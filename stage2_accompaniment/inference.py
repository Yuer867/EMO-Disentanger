import os
import sys
import time
import yaml
import shutil
import argparse
import numpy as np
from itertools import chain
from collections import defaultdict
import torch

from dataloader import REMISkylineToMidiTransformerDataset, pickle_load
from model.music_performer import MusicPerformer
from convert2midi import event_to_midi
from convert_key import degree2pitch, roman2majorDegree, roman2minorDegree

sys.path.append('./model')

max_bars = 128
max_dec_inp_len = 2048

temp, top_p = 1.1, 0.99
emotion_events = ['Emotion_Q1', 'Emotion_Q2', 'Emotion_Q3', 'Emotion_Q4']
samp_per_piece = 1

major_map = [0, 4, 7]
minor_map = [0, 3, 7]
diminished_map = [0, 3, 6]
augmented_map = [0, 4, 8]
dominant_map = [0, 4, 7, 10]
major_seventh_map = [0, 4, 7, 11]
minor_seventh_map = [0, 3, 7, 10]
diminished_seventh_map = [0, 3, 6, 9]
half_diminished_seventh_map = [0, 3, 6, 10]
sus_2_map = [0, 2, 7]
sus_4_map = [0, 5, 7]

chord_maps = {
    'M': major_map,
    'm': minor_map,
    'o': diminished_map,
    '+': augmented_map,
    '7': dominant_map,
    'M7': major_seventh_map,
    'm7': minor_seventh_map,
    'o7': diminished_seventh_map,
    '/o7': half_diminished_seventh_map,
    'sus2': sus_2_map,
    'sus4': sus_4_map
}
chord_maps = {k: np.array(v) for k, v in chord_maps.items()}

DEFAULT_SCALE = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
MAJOR_KEY = np.array(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
MINOR_KEY = np.array(['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b'])


###############################################
# sampling utilities
###############################################
def construct_inadmissible_set(tempo_val, event2idx, tolerance=20):
    inadmissibles = []

    for k, i in event2idx.items():
        if 'Tempo' in k and 'Conti' not in k and abs(int(k.split('_')[-1]) - tempo_val) > tolerance:
            inadmissibles.append(i)

    print(inadmissibles)

    return np.array(inadmissibles)


def temperature(logits, temperature, inadmissibles=12):
    if inadmissibles is not None:
        logits[inadmissibles] -= np.inf

    try:
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        assert np.count_nonzero(np.isnan(probs)) == 0
    except:
        print('overflow detected, use 128-bit')
        logits = logits.astype(np.float128)
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        probs = probs.astype(float)
    return probs


def nucleus(probs, p):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][1]
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:3]  # just assign a value
    candi_probs = np.array([probs[i] for i in candi_index], dtype=np.float64)
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word


##############################################
# data manipulation utilities
##############################################
def merge_tracks(melody_track, chord_track):
    events = melody_track[1:3]

    melody_beat = defaultdict(list)
    if len(melody_track) > 3:
        note_seq = []
        beat = melody_track[3]
        melody_track = melody_track[4:]
        for p in range(len(melody_track)):
            if 'Beat' in melody_track[p]:
                melody_beat[beat] = note_seq
                note_seq = []
                beat = melody_track[p]
            else:
                note_seq.append(melody_track[p])
        melody_beat[beat] = note_seq

    chord_beat = defaultdict(list)
    if len(chord_track) > 2:
        chord_seq = []
        beat = chord_track[2]
        chord_track = chord_track[3:]
        for p in range(len(chord_track)):
            if 'Beat' in chord_track[p]:
                chord_beat[beat] = chord_seq
                chord_seq = []
                beat = chord_track[p]
            else:
                chord_seq.append(chord_track[p])
        chord_beat[beat] = chord_seq

    for b in range(16):
        beat = 'Beat_{}'.format(b)
        if beat in chord_beat or beat in melody_beat:
            events.append(beat)
            if beat in chord_beat:
                events.extend(chord_beat[beat])
            if beat in melody_beat:
                events.extend(melody_beat[beat])

    return events


def read_generated_events(events_file, event2idx):
    events = open(events_file).read().splitlines()
    if 'Key' in events[0]:
        key = events[0]
    else:
        key = 'Key_C'

    bar_pos = np.where(np.array(events) == 'Bar_None')[0].tolist()
    bar_pos.append(len(events))

    lead_sheet_bars = []
    for b in range(len(bar_pos)-1):
        lead_sheet_bars.append(events[bar_pos[b]: bar_pos[b+1]])

    for bar in range(len(lead_sheet_bars)):
        lead_sheet_bars[bar] = [event2idx[e] for e in lead_sheet_bars[bar]]

    return key, lead_sheet_bars


def word2event(word_seq, idx2event):
    return [idx2event[w] for w in word_seq]


def extract_skyline_from_val_data(val_input, idx2event, event2idx):
    tempo = val_input[0]

    skyline_starts = np.where(val_input == event2idx['Track_Skyline'])[0].tolist()
    midi_starts = np.where(val_input == event2idx['Track_Midi'])[0].tolist()

    assert len(skyline_starts) == len(midi_starts)

    skyline_bars = []
    for st, ed in zip(skyline_starts, midi_starts):
        bar_skyline_events = val_input[st + 1: ed].tolist()
        skyline_bars.append(bar_skyline_events)

    return tempo, skyline_bars


def extract_midi_events_from_generation(key, events, relative_melody=False):
    if relative_melody:
        new_events = []
        keyname = key.split('_')[1]
        for evs in events:
            if 'Note_Octave' in evs:
                octave = int(evs.split('_')[2])
            elif 'Note_Degree' in evs:
                roman = evs.split('_')[2]
                pitch = degree2pitch(keyname, octave, roman)
                pitch = max(21, pitch)
                pitch = min(108, pitch)
                if pitch < 21 or pitch > 108:
                    raise ValueError('Pitch value must be in (21, 108), but gets {}'.format(pitch))
                new_events.append('Note_Pitch_{}'.format(pitch))
            elif 'Chord_' in evs:
                if 'None' in evs or 'Conti' in evs:
                    new_events.append(evs)
                else:
                    root, quality = evs.split('_')[1], evs.split('_')[2]
                    if keyname in MAJOR_KEY:
                        root = roman2majorDegree[root]
                    else:
                        root = roman2minorDegree[root]
                    new_events.append('Chord_{}_{}'.format(root, quality))
            else:
                new_events.append(evs)
        events = new_events

    lead_sheet_starts = np.where(np.array(events) == 'Track_LeadSheet')[0].tolist()
    full_starts = np.where(np.array(events) == 'Track_Full')[0].tolist()

    midi_bars = []
    for st, ed in zip(full_starts, lead_sheet_starts[1:] + [len(events)]):
        bar_midi_events = events[st + 1: ed]
        midi_bars.append(bar_midi_events)

    return midi_bars


def get_position_idx(event):
    return int(event.split('_')[-1])


def event_to_txt(events, output_event_path):
    f = open(output_event_path, 'w')
    print(*events, sep='\n', file=f)


def midi_to_wav(midi_path, output_path):
    sound_font_path = 'SalamanderGrandPiano-SF2-V3+20200602/SalamanderGrandPiano-V3+20200602.sf2'
    fs = FluidSynth(sound_font_path)
    fs.midi_to_audio(midi_path, output_path)


################################################
# main generation function
################################################
def generate_conditional(model, event2idx, idx2event, lead_sheet_events, primer,
                         max_events=10000, skip_check=False, max_bars=None,
                         temp=1.2, top_p=0.9, inadmissibles=None):
    generated = primer + [event2idx['Track_LeadSheet']] + lead_sheet_events[0] + [event2idx['Track_Full']]
    # print(generated)
    seg_inp = [0 for _ in range(len(generated))]
    seg_inp[-1] = 1
    # print(seg_inp)

    target_bars, generated_bars = len(lead_sheet_events), 0
    if max_bars is not None:
        target_bars = min(max_bars, target_bars)

    steps = 0
    time_st = time.time()
    cur_pos = 0
    failed_cnt = 0

    while generated_bars < target_bars:
        assert len(generated) == len(seg_inp)
        if len(generated) < max_dec_inp_len:
            dec_input = torch.tensor([generated]).long().to(next(model.parameters()).device)
            dec_seg_inp = torch.tensor([seg_inp]).long().to(next(model.parameters()).device)
        else:
            dec_input = torch.tensor([generated[-max_dec_inp_len:]]).long().to(next(model.parameters()).device)
            dec_seg_inp = torch.tensor([seg_inp[-max_dec_inp_len:]]).long().to(next(model.parameters()).device)

        # sampling
        logits = model(
            dec_input,
            seg_inp=dec_seg_inp,
            keep_last_only=True,
            attn_kwargs={'omit_feature_map_draw': steps > 0}
        )
        logits = (logits[0]).cpu().detach().numpy()
        probs = temperature(logits, temp, inadmissibles=inadmissibles)
        word = nucleus(probs, top_p)
        word_event = idx2event[word]

        if not skip_check:
            if 'Beat' in word_event:
                event_pos = get_position_idx(word_event)
                if not event_pos >= cur_pos:
                    failed_cnt += 1
                    print('[info] position not increasing, failed cnt:', failed_cnt)
                    if failed_cnt >= 256:
                        print('[FATAL] model stuck, exiting with generated events ...')
                        return generated
                    continue
                else:
                    cur_pos = event_pos
                    failed_cnt = 0

        if word_event == 'Track_LeadSheet':
            steps += 1
            generated.append(word)
            seg_inp.append(0)
            generated_bars += 1
            print('[info] generated {} bars, #events = {}'.format(generated_bars, len(generated)))

            if generated_bars < target_bars:
                generated.extend(lead_sheet_events[generated_bars])
                seg_inp.extend([0 for _ in range(len(lead_sheet_events[generated_bars]))])

                generated.append(event2idx['Track_Full'])
                seg_inp.append(1)
                cur_pos = 0
            continue

        if word_event == 'PAD_None' or (word_event == 'EOS_None' and generated_bars < target_bars - 1):
            continue
        elif word_event == 'EOS_None' and generated_bars == target_bars - 1:
            print('[info] gotten eos')
            generated.append(word)
            break

        generated.append(word)
        seg_inp.append(1)
        steps += 1

        if len(generated) > max_events:
            print('[info] max events reached')
            break

    print('-- generated events:', len(generated))
    print('-- time elapsed  : {:.2f} secs'.format(time.time() - time_st))
    print('-- time per event: {:.2f} secs'.format((time.time() - time_st) / len(generated)))
    return generated[:-1]


if __name__ == '__main__':
    # configuration
    parser = argparse.ArgumentParser(description='')
    required = parser.add_argument_group('required arguments')
    required.add_argument('-c', '--configuration',
                          choices=['stage2_accompaniment/config/pop1k7_pretrain.yaml',
                                   'stage2_accompaniment/config/emopia_finetune.yaml'],
                          help='configurations of training', required=True)
    required.add_argument('-r', '--representation',
                          choices=['absolute', 'functional'],
                          help='representation for symbolic music', required=True)
    parser.add_argument('-i', '--inference_params',
                        default='best_weight/Functional-two/emopia_acccompaniment_finetune/ep300_loss0.338_params.pt',
                        help='inference parameters')
    parser.add_argument('-o', '--output_dir',
                        default='generation/emopia_functional_two',
                        help='output directory')
    parser.add_argument('-p', '--play_midi',
                        default=False,
                        help='play midi to audio using FluidSynth', action='store_true')
    args = parser.parse_args()

    train_conf_path = args.configuration
    train_conf = yaml.load(open(train_conf_path, 'r'), Loader=yaml.FullLoader)
    print(train_conf)

    representation = args.representatiossn
    if representation == 'absolute':
        relative_melody = False
    elif representation == 'functional':
        relative_melody = True

    inference_param_path = args.inference_params
    gen_leadsheet_dir = args.output_dir
    play_midi = args.play_midi

    train_conf_ = train_conf['training']
    gpuid = train_conf_['gpuid']
    torch.cuda.set_device(gpuid)

    val_split = train_conf['data_loader']['val_split']
    dset = REMISkylineToMidiTransformerDataset(
        train_conf['data_loader']['data_path'].format(representation),
        train_conf['data_loader']['vocab_path'].format(representation),
        model_dec_seqlen=train_conf['model']['max_len'],
        pieces=pickle_load(val_split),
        pad_to_same=True,
    )

    model_conf = train_conf['model']
    model = MusicPerformer(
        dset.vocab_size, model_conf['n_layer'], model_conf['n_head'],
        model_conf['d_model'], model_conf['d_ff'], model_conf['d_embed'],
        use_segment_emb=model_conf['use_segemb'], n_segment_types=model_conf['n_segment_types'],
        favor_feature_dims=model_conf['feature_map']['n_dims']
    ).cuda()

    pretrained_dict = torch.load(inference_param_path, map_location='cpu')
    pretrained_dict = {
        k: v for k, v in pretrained_dict.items() if 'feature_map.omega' not in k
    }
    model_state_dict = model.state_dict()
    model_state_dict.update(pretrained_dict)
    model.load_state_dict(model_state_dict)

    model.eval()
    print('[info] model loaded')

    shutil.copy(train_conf_path, os.path.join(gen_leadsheet_dir, 'config_full.yaml'))

    lead_sheet_files = [x for x in os.listdir(gen_leadsheet_dir)]
    print('[# pieces]', len(lead_sheet_files))

    if representation == 'functional':
        files = [os.path.join(gen_leadsheet_dir, i) for i in os.listdir(gen_leadsheet_dir) if 'roman.txt' in i]
    elif representation in ['absolute', 'key']:
        files = [os.path.join(gen_leadsheet_dir, i) for i in os.listdir(gen_leadsheet_dir) if '.txt' in i]

    for file in files:
        out_name = '_'.join(file.split('/')[-1].split('_')[:2])
        print(file)
        if 'Positive' in file:
            emotion_candidate = ['Q1', 'Q4']
        elif 'Negative' in file:
            emotion_candidate = ['Q2', 'Q3']
        elif 'Q1' in file:
            emotion_candidate = ['Q1']
        elif 'Q2' in file:
            emotion_candidate = ['Q2']
        elif 'Q3' in file:
            emotion_candidate = ['Q3']
        elif 'Q4' in file:
            emotion_candidate = ['Q4']
        elif 'None' in file:
            emotion_candidate = ['None']
        else:
            raise ValueError('wrong emotion label')

        for e in emotion_candidate:
            if os.path.exists(os.path.join(gen_leadsheet_dir, out_name + '_' + e + '_full.mid')):
                print('[info] {} exists, skipping ...'.format(os.path.join(gen_leadsheet_dir, out_name + '_' + e + '_full.mid')))
                continue
            print(e)
            emotion = dset.event2idx['Emotion_{}'.format(e)]
            tempo = dset.event2idx['Tempo_{}'.format(110)]
            key, lead_sheet_events = read_generated_events(file, dset.event2idx)
            print(key)
            if representation in ['functional', 'key']:
                primer = [emotion, dset.event2idx[key], tempo]
            elif representation == 'absolute':
                primer = [emotion, tempo]

            with torch.no_grad():
                generated = generate_conditional(model, dset.event2idx, dset.idx2event,
                                                 lead_sheet_events, primer=primer,
                                                 max_bars=max_bars, temp=temp, top_p=top_p,
                                                 inadmissibles=None)

            generated = word2event(generated, dset.idx2event)
            generated = extract_midi_events_from_generation(key, generated, relative_melody=relative_melody)

            output_midi_path = os.path.join(gen_leadsheet_dir, out_name + '_' + e + '_full.mid')
            event_to_midi(
                key,
                list(chain(*generated[:max_bars])),
                mode='full',
                output_midi_path=output_midi_path
            )

            if play_midi:
                from midi2audio import FluidSynth

                output_wav_path = os.path.join(gen_leadsheet_dir, out_name + '_' + e + '_full.wav')
                midi_to_wav(output_midi_path, output_wav_path)
