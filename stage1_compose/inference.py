import sys
import os
import random
import pickle5 as pickle
import argparse

import yaml
import torch
import shutil
import numpy as np

from model.plain_transformer import PlainTransformer
from convert2midi import event_to_midi, TempoEvent
from utils import pickle_load
from inference_utils import generate_plain_xl
from convert_key import degree2pitch, roman2majorDegree, roman2minorDegree, MAJOR_KEY

sys.path.append('./model/')
sys.path.append('./')


def read_vocab(vocab_file):
    event2idx, idx2event = pickle_load(vocab_file)
    orig_vocab_size = len(event2idx)
    pad_token = orig_vocab_size
    event2idx['PAD_None'] = pad_token
    vocab_size = pad_token + 1

    return event2idx, idx2event, vocab_size


def get_leadsheet_prompt(data_dir, piece, prompt_n_bars):
    bar_pos, evs = pickle_load(os.path.join(data_dir, piece))

    prompt_evs = [
        '{}_{}'.format(x['name'], x['value']) for x in evs[: bar_pos[prompt_n_bars] + 1]
    ]
    assert len(np.where(np.array(prompt_evs) == 'Bar_None')[0]) == prompt_n_bars + 1
    target_bars = len(bar_pos)

    return prompt_evs, target_bars


def relative2absolute(key, events):
    new_events = []
    key = key.split('_')[1]
    for evs in events:
        if 'Note_Octave' in evs:
            octave = int(evs.split('_')[2])
        elif 'Note_Degree' in evs:
            roman = evs.split('_')[2]
            pitch = degree2pitch(key, octave, roman)
            pitch = max(21, pitch)
            pitch = min(108, pitch)
            if pitch < 21 or pitch > 108:
                raise ValueError('Pitch value must be in (21, 108), but gets {}'.format(pitch))
            new_events.append('Note_Pitch_{}'.format(pitch))
        elif 'Chord_' in evs:
            if 'None' in evs:
                new_events.append(evs)
            else:
                root, quality = evs.split('_')[1], evs.split('_')[2]
                if key in MAJOR_KEY:
                    root = roman2majorDegree[root]
                else:
                    root = roman2minorDegree[root]
                new_events.append('Chord_{}_{}'.format(root, quality))
        else:
            new_events.append(evs)
    events = new_events

    return events


def event_to_txt(events, output_event_path):
    f = open(output_event_path, 'w')
    print(*events, sep='\n', file=f)


def midi_to_wav(midi_path, output_path):
    sound_font_path = 'SalamanderGrandPiano-SF2-V3+20200602/SalamanderGrandPiano-V3+20200602.sf2'
    fs = FluidSynth(sound_font_path)
    fs.midi_to_audio(midi_path, output_path)


if __name__ == '__main__':
    # configuration
    parser = argparse.ArgumentParser(description='')
    required = parser.add_argument_group('required arguments')
    required.add_argument('-c', '--configuration',
                          choices=['stage1_compose/config/hooktheory_pretrain.yaml',
                                   'stage1_compose/config/emopia_finetune.yaml',
                                   'stage1_compose/config/pop1k7_pretrain.yaml',
                                   'stage1_compose/config/emopia_finetune_full.yaml'],
                          help='configurations of training', required=True)
    required.add_argument('-r', '--representation',
                          choices=['remi', 'functional'],
                          help='representation for symbolic music', required=True)
    required.add_argument('-m', '--mode',
                          choices=['lead_sheet', 'full_song'],
                          help='generation mode', required=True)
    parser.add_argument('-i', '--inference_params',
                        default='best_weight/Functional-two/emopia_lead_sheet_finetune/ep016_loss0.685_params.pt',
                        help='inference parameters')
    parser.add_argument('-o', '--output_dir',
                        default='generation/emopia_functional_two',
                        help='output directory')
    parser.add_argument('-p', '--play_midi',
                        default=False,
                        help='play midi to audio using FluidSynth', action='store_true')
    parser.add_argument('-n', '--n_groups',
                        default=20,
                        help='number of groups to generate')
    args = parser.parse_args()

    config_path = args.configuration
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    representation = args.representation
    mode = args.mode
    inference_params = args.inference_params
    out_dir = args.output_dir
    play_midi = args.play_midi
    n_groups = int(args.n_groups)
    key_determine = 'rule'
    print('representation: {}, key determine: {}'.format(representation, key_determine))

    max_bars = 128
    if mode == 'lead_sheet':
        temp = 1.2
        top_p = 0.97
        max_dec_len = 512
        emotions = ['Positive', 'Negative']
    elif mode == 'full_song':
        temp = 1.1
        top_p = 0.99
        max_dec_len = 2400
        emotions = ['Q1', 'Q2', 'Q3', 'Q4']
    print('[nucleus parameters] t = {}, p = {}'.format(temp, top_p))

    torch.cuda.device(config['device'])

    # for generation w/ melody prompts
    use_prompt = False
    prompt_bars = 8

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    event2idx, idx2event, vocab_size = read_vocab(config['data']['vocab_path'].format(representation))

    if use_prompt:
        prompt_pieces = pickle_load(config['data']['val_split'])
        prompt_pieces = [x for x in prompt_pieces if os.path.exists(
            os.path.join(config['data']['data_dir'].format(representation), x)
        )]
        if len(prompt_pieces) > n_groups:
            prompt_pieces = random.sample(prompt_pieces, n_groups)

        pickle.dump(
            prompt_pieces,
            open(os.path.join(out_dir, 'sampled_pieces.pkl'), 'wb')
        )
        prompts = []
        for p in prompt_pieces:
            prompts.append(
                get_leadsheet_prompt(
                    config['data']['data_dir'], p,
                    prompt_bars
                )
            )

    mconf = config['model']
    model = PlainTransformer(
        mconf['d_word_embed'],
        vocab_size,
        mconf['decoder']['n_layer'],
        mconf['decoder']['n_head'],
        mconf['decoder']['d_model'],
        mconf['decoder']['d_ff'],
        mconf['decoder']['tgt_len'],
        mconf['decoder']['tgt_len'],
        dec_dropout=mconf['decoder']['dropout'],
        pre_lnorm=mconf['pre_lnorm']
    ).cuda()
    print('[info] # params:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    pretrained_dict = torch.load(inference_params, map_location='cpu')
    model.load_state_dict(pretrained_dict)
    model.eval()

    if mode == 'lead_sheet':
        shutil.copy(config_path, os.path.join(out_dir, 'config_lead.yaml'))
    elif mode == 'full_song':
        shutil.copy(config_path, os.path.join(out_dir, 'config_full.yaml'))

    generated_pieces = 0
    total_pieces = n_groups
    gen_times = []

    while generated_pieces < n_groups:
        for emotion in emotions:
            out_name = 'samp_{:02d}_{}'.format(generated_pieces, emotion)
            print(out_name)
            if os.path.exists(os.path.join(out_dir, out_name + '.mid')):
                print('[info] {} exists, skipping ...'.format(out_name))
                continue

            if not use_prompt:
                # tempo_range = range(65, 165, 3)
                # tempo = random.choice(tempo_range)
                tempo = 110
                orig_tempos = [TempoEvent(tempo, 0, 0)]
                print('[global tempo]', orig_tempos[0].tempo)
            else:
                # target_bars = prompts[p][1]
                target_bars = prompts[generated_pieces][1]
                tempo = 110
                orig_tempos = [TempoEvent(tempo, 0, 0)]
                # orig_tempos = [
                #   TempoEvent(int(prompts[generated_pieces][0][0].split('_')[-1]), 0, 0)
                # ]

            # print(' -- generating leadsheet #{} of {}'.format(
            #     generated_pieces + 1, total_pieces
            # ))

            if not use_prompt:
                gen_words, t_sec = generate_plain_xl(
                                      model,
                                      event2idx, idx2event,
                                      max_events=max_dec_len, max_bars=max_bars,
                                      # primer=['Tempo_{}'.format(orig_tempos[0].tempo), 'Bar_None'],
                                      primer=['Emotion_{}'.format(emotion)],
                                      temp=temp, top_p=top_p,
                                      representation=representation,
                                      key_determine=key_determine
                                    )
            else:
                gen_words, t_sec = generate_plain_xl(
                                      model,
                                      event2idx, idx2event,
                                      max_events=max_dec_len, max_bars=target_bars,
                                      # primer=prompts[p][0],
                                      primer=['Emotion_{}'.format(emotion)] + prompts[generated_pieces][0][1:],
                                      temp=temp, top_p=top_p,
                                      prompt_bars=prompt_bars,
                                      representation=representation,
                                      key_determine=key_determine
                                    )

            gen_words = [idx2event[w] for w in gen_words]

            key = None
            for evs in gen_words:
                if 'Key' in evs:
                    key = evs
            if key is None:
                # raise ValueError('invalid key')
                key = 'Key_C'

            if representation == 'functional':
                gen_words_roman = gen_words[1:]
                gen_words = relative2absolute(key, gen_words)[1:]
            else:
                gen_words = gen_words[1:]

            if gen_words is None:  # model failed repeatedly
                continue
            # if len(gen_words) >= max_dec_len:
            #     continue
            # if len(np.where(np.array(gen_words) == event2idx['Bar_None'])[0]) >= max_bars:
            #     continue

            if mode == 'lead_sheet':
                event_to_midi(key, gen_words, mode=mode,
                              output_midi_path=os.path.join(out_dir, out_name + '.mid'),
                              play_chords=True, enforce_tempo=True, enforce_tempo_evs=orig_tempos)
            elif mode == 'full_song':
                event_to_midi(key, gen_words, mode=mode,
                              output_midi_path=os.path.join(out_dir, out_name + '.mid'))
            event_to_txt(gen_words, output_event_path=os.path.join(out_dir, out_name + '.txt'))
            if representation == 'functional':
                event_to_txt(gen_words_roman, output_event_path=os.path.join(out_dir, out_name + '_roman.txt'))

            gen_times.append(t_sec)

            if play_midi:
                from midi2audio import FluidSynth

                output_midi_path = os.path.join(out_dir, out_name + '.mid')
                output_wav_path = os.path.join(out_dir, out_name + '.wav')
                midi_to_wav(output_midi_path, output_wav_path)

        generated_pieces += 1

    print('[info] finished generating {} pieces, avg. time: {:.2f} +/- {:.2f} secs.'.format(
        generated_pieces, np.mean(gen_times), np.std(gen_times)
    ))
