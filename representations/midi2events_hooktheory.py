import os
import gzip
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm

import miditoolkit
from miditoolkit.midi import containers

from midi2events_emopia import midi2corpus, corpus2lead
from convert_key import find_key_hooktheory

seventh = [[4, 3, 3], [4, 3, 4], [3, 4, 3], [3, 3, 3], [3, 3, 4]]
triad = [[4, 3], [3, 4], [3, 3], [4, 4], [2, 5], [5, 2]]
interval2symbol = {'433': '7', '434': 'M7', '343': 'm7', '333': 'o7', '334': '/o7',
                   '43': 'M', '34': 'm', '33': 'o', '44': '+',
                   '25': 'sus2', '52': 'sus4'}

BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4
DEFAULT_TEMPO = 110
MELODY_OCTAVE = 5  # mean pitch of EMOPIA is 72
VELOCITY = 100

MAJOR_KEY = np.array(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
MINOR_KEY = np.array(['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b'])
IDX_TO_KEY = {
    9: 'A',
    10: 'A#',
    11: 'B',
    0: 'C',
    1: 'C#',
    2: 'D',
    3: 'D#',
    4: 'E',
    5: 'F',
    6: 'F#',
    7: 'G',
    8: 'G#'
}
KEY_TO_IDX = {
    v: k for k, v in IDX_TO_KEY.items()
}


def chord_mhot(interval):
    mhot = np.zeros(12, dtype=int)
    interval = np.cumsum([0] + interval)
    for i in interval:
        mhot[i % 12] = 1
    # print(interval, mhot)
    return mhot


def chord_simplify(interval, invert=True):
    """
    simplify into 11 chord qualities
    """
    # belong to default eleven classes
    if interval in seventh + triad:
        return interval

    mhot = chord_mhot(interval)
    # simplify to seventh
    if interval[:3] in seventh:
        return interval[:3]
    if invert:
        for c in seventh:
            if (mhot & chord_mhot(c) == chord_mhot(c)).all():
                return c

    # simplify to triad
    if interval[:2] in triad:
        return interval[:2]
    if invert:
        for c in triad:
            if (mhot & chord_mhot(c) == chord_mhot(c)).all():
                return c

    # add a fifth note
    mhot[7] = 1
    for c in seventh:
        if (mhot & chord_mhot(c) == chord_mhot(c)).all():
            return c
    for c in triad:
        if (mhot & chord_mhot(c) == chord_mhot(c)).all():
            return c

    return False


def annotation2midi(annotations, relative_chord=False, transpose_to_C=False):
    """
    convert annotations to midi files
    """
    midi_obj = miditoolkit.midi.parser.MidiFile()
    midi_obj.time_signature_changes.append(containers.TimeSignature(numerator=4, denominator=4, time=0))
    midi_obj.instruments.append(containers.Instrument(program=0, is_drum=False, name='piano'))

    # find key, mode and scale degree
    key = IDX_TO_KEY[annotations['keys'][0]['tonic_pitch_class']]
    mode = list2str(annotations['keys'][0]['scale_degree_intervals'])
    if mode == '212212':
        keyname = key.lower()
    else:
        keyname = key.upper()

    # whether transpose to C major/minor
    if transpose_to_C:
        if KEY_TO_IDX[key] >= 6:
            pitch_offset = 12 - KEY_TO_IDX[key]
        else:
            pitch_offset = - KEY_TO_IDX[key]
        root2degree = {MAJOR_KEY[i]: str(i) for i in range(len(MAJOR_KEY))}
    elif relative_chord:
        pitch_offset = 0
        start = np.where(MAJOR_KEY == key)[0][0]
        scale_range = np.concatenate([MAJOR_KEY[start:], MAJOR_KEY[:start]], axis=0)
        root2degree = {scale_range[i]: str(i) for i in range(len(scale_range))}
    else:
        pitch_offset = 0
        root2degree = {MAJOR_KEY[i]: str(i) for i in range(len(MAJOR_KEY))}

    # add key
    midi_obj.markers.append(containers.Marker(text='global_key_' + keyname, time=0))

    # add tempo
    midi_obj.tempo_changes.append(containers.TempoChange(tempo=DEFAULT_TEMPO, time=0))
    midi_obj.markers.append(containers.Marker(text='global_bpm_' + str(DEFAULT_TEMPO), time=0))

    # add notes
    melody = annotations['melody']
    for note in melody:
        onset = int(note['onset'] * BEAT_RESOL)
        offset = int(note['offset'] * BEAT_RESOL)
        if onset == offset:
            continue

        pitch = note['pitch_class'] + (MELODY_OCTAVE + note['octave']) * 12 + pitch_offset
        midi_note = containers.Note(start=onset, end=offset, pitch=pitch, velocity=VELOCITY)
        midi_obj.instruments[0].notes.append(midi_note)

    # add max_tick
    max_tick = max([note.end for note in midi_obj.instruments[0].notes])
    midi_obj.max_tick = max_tick

    # add chord
    harmony = annotations['harmony']
    dedup_chords = []
    for chord in harmony:

        # time
        onset = int(np.round(chord['onset']) * BEAT_RESOL)
        offset = int(np.round(chord['offset']) * BEAT_RESOL)
        max_tick = max(max_tick, offset)
        if onset == offset:
            continue

        # root, quality
        root = IDX_TO_KEY[(chord['root_pitch_class'] + pitch_offset) % 12]
        interval = chord_simplify(chord['root_position_intervals'], invert=True)
        if interval:
            quality = interval2symbol[''.join(str(i) for i in interval)]
            dedup_chords.append(containers.Marker(time=onset, text=root + '_' + quality + '_' + root))
        else:
            dedup_chords.append(containers.Marker(time=onset, text='None_None_None'))
    dedup_chords.sort(key=lambda x: x.time)

    # repeat chord
    chords = []
    beat2chord = {c.time: c for c in dedup_chords}
    prev_chord = 'None_None_None'
    max_beat = int(np.ceil(max_tick / BEAT_RESOL) * BEAT_RESOL)
    for beat in range(0, max_beat, BEAT_RESOL):
        if beat in beat2chord:
            chords.append(beat2chord[beat])
            prev_chord = beat2chord[beat].text
        else:
            chords.append(containers.Marker(time=beat, text=prev_chord))

    # translate chord label to scale degree
    trans_chords = []
    for c in chords:
        if 'None' in c.text or 'Conti' in c.text:
            trans_chords.append(c)
        else:
            root = root2degree[c.text.split('_')[0]]
            quality = c.text.split('_')[1]
            bass = root2degree[c.text.split('_')[2]]
            text = '_'.join([root, quality, bass])
            trans_chords.append(containers.Marker(time=c.time, text=text))
    assert len(trans_chords) == len(chords)
    chords = trans_chords

    midi_obj.markers += chords

    return midi_obj


def list2str(a_list):
    return ''.join([str(i) for i in a_list])


if __name__ == '__main__':
    """
    convert midi to events
    """
    # configuration
    parser = argparse.ArgumentParser(description='')
    required = parser.add_argument_group('required arguments')
    required.add_argument('-r', '--representation',
                          choices=['absolute', 'functional'],
                          help='representation for symbolic music', required=True)
    parser.add_argument('-e', '--num_emotion', default=2, help='number of emotion types')
    args = parser.parse_args()
    representation = args.representation
    num_emotion = args.num_emotion

    if representation == 'absolute':
        transpose_to_C, relative_chord, relative_melody = False, False, False
    elif representation == 'functional':
        transpose_to_C, relative_chord, relative_melody = False, True, True
    else:
        raise ValueError("invalid representation {}, choose from [absolute, functional]"
                         .format(representation))
    print('whether transpose_to_C: {}, whether relative_chord: {}, whether relative_melody: {}'.
          format(transpose_to_C, relative_chord, relative_melody))
    if relative_chord and transpose_to_C:
        raise ValueError("'relative_chord' and 'transpose_to_C' can't be True together")

    # read json data
    print('read and filter available clips ...')
    data_home = 'midi_data/HookTheory'
    with gzip.open(os.path.join(data_home, 'Hooktheory.json.gz'), 'r') as f:
        dataset = json.load(f)

    # get available clips for pre-training
    available_set = {k: v for k, v in dataset.items()
                     if 'MELODY' in v['tags']
                     and 'HARMONY' in v['tags']
                     and 'METER_CHANGES' not in v['tags']
                     and v['annotations']['meters'][0]['beats_per_bar'] == 4
                     and v['annotations']['meters'][0]['beat_unit'] == 4
                     and 'KEY_CHANGES' not in v['tags']
                     and list2str(v['annotations']['keys'][0]['scale_degree_intervals']) in ['221222', '212212']}
    print('# available clips:', len(available_set))

    # convert clips to midi files
    print('convert clips to midi ...')
    midi_home = os.path.join(data_home, 'midis_chord11_{}'.format(representation))
    os.makedirs(midi_home, exist_ok=True)
    print('midi dir:', midi_home)

    for k, v in tqdm(available_set.items()):
        clip_name = k + '.mid'
        output_path = os.path.join(midi_home, clip_name)

        annotations = v['annotations']
        midi_obj = annotation2midi(annotations, relative_chord=relative_chord, transpose_to_C=transpose_to_C)
        midi_obj.dump(filename=output_path)

    midi_files = os.listdir(midi_home)
    print('# midi files: ', len(midi_files))

    # convert midi to pkl
    print('convert midi to lead sheet ...')
    lead_sheet_events_dir = 'events/stage1/hooktheory_events/lead_sheet_chord11_{}/events'.format(representation)
    os.makedirs(lead_sheet_events_dir, exist_ok=True)
    print('save dir:', lead_sheet_events_dir)

    # load dict for key
    clip2keyname, clip2keymode = find_key_hooktheory()

    num_sample = 0
    for file in tqdm(midi_files):
        midi_path = os.path.join(midi_home, file)
        filename = file[:-4]
        lead_midi_obj = miditoolkit.midi.parser.MidiFile(midi_path)

        # get emotion tag
        emotion = None

        # convert midi to lead sheet
        lead_data = midi2corpus(lead_midi_obj)
        pos, events = corpus2lead(lead_data, emotion, relative_chord=relative_chord, relative_melody=relative_melody)
        if len(pos) < 4:
            continue
        num_sample += 1

        # save
        pickle.dump((pos, events), open(os.path.join(lead_sheet_events_dir, filename + '.pkl'), 'wb'))

    print('# samples:', num_sample)
