import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from utils import json_read, pickle_dump
from convert_key import MAJOR_KEY, IDX_TO_KEY, minorDegree2roman, majorDegree2roman, pitch2degree


def create_event(name, value):
    event = dict()
    event['name'] = name
    event['value'] = value
    return event


def event2lead_full(events, keyname, relative_chord, relative_melody):
    functional_events = []
    ls_start = []
    full_start = []

    # --- emotion --- #
    emotion_event = create_event('Emotion', None)
    functional_events.append(emotion_event)

    # --- key --- #
    if relative_chord:
        key_event = create_event('Key', keyname)
        functional_events.append(key_event)

    # translate chord label to scale degree according to key
    root2degree = {MAJOR_KEY[i]: str(i) for i in range(len(MAJOR_KEY))}
    if relative_chord:
        start = np.where(MAJOR_KEY == keyname.upper())[0][0]
        scale_range = np.concatenate([MAJOR_KEY[start:], MAJOR_KEY[:start]], axis=0)
        root2degree = {scale_range[i]: str(i) for i in range(len(scale_range))}

    for evs in events:
        # --- relative chord --- #
        if evs['name'] == 'Chord' and evs['value'] not in ['Conti_Conti', 'None_None']:
            root, quality = evs['value'].split('_')
            root = root2degree[IDX_TO_KEY[int(root)]]
            if relative_melody and root != 'None':
                if keyname in MAJOR_KEY:
                    root = majorDegree2roman[int(root)]
                else:
                    root = minorDegree2roman[int(root)]
            functional_events.append(create_event('Chord', root + '_' + quality))

        # --- relative melody --- #
        elif evs['name'] == 'Note_Pitch':
            if relative_melody:
                pitch = evs['value']
                octave, roman = pitch2degree(keyname, pitch)
                functional_events.append(create_event('Note_Octave', octave))
                functional_events.append(create_event('Note_Degree', roman))
            else:
                functional_events.append(evs)
        elif evs['name'] == 'Track':
            if evs['value'] == 'Skyline':
                ls_start.append(len(functional_events))
                functional_events.append(create_event('Track', 'LeadSheet'))
            elif evs['value'] == 'Midi':
                full_start.append(len(functional_events))
                functional_events.append(create_event('Track', 'Full'))

        # --- track, bar, beat, tempo, EOS --- #
        else:
            functional_events.append(evs)

    ls_start.append(len(functional_events))
    assert len(ls_start) == len(full_start) + 1

    # --- add chord event for Beat_0/4/8/12 --- #
    final_events = functional_events[:ls_start[0]]
    ls_start_new = []
    full_start_new = []
    chord = 'None_None'
    for s in range(len(full_start)):
        ls_start_new.append(len(final_events))
        lead_sheet_events = functional_events[ls_start[s]:full_start[s]]
        full_song_events = functional_events[full_start[s]:ls_start[s + 1]]
        new_events = lead_sheet_events[:2]
        beat_seq = {}
        if len(lead_sheet_events) > 3:
            for evs in lead_sheet_events[2:]:
                if evs['name'] == 'Beat':
                    beat = evs['value']
                    beat_seq[beat] = []
                else:
                    if beat not in beat_seq:
                        break
                    beat_seq[beat].append(evs)
        for b in [0, 4, 8, 12]:
            if b not in beat_seq:
                beat_seq[b] = [create_event('Chord', chord)]
            else:
                if beat_seq[b][0]['name'] == 'Chord':
                    chord = beat_seq[b][0]['value']
                else:
                    beat_seq[b] = [create_event('Chord', chord)] + beat_seq[b]
        assert beat_seq[0][0]['name'] == 'Chord'
        assert beat_seq[4][0]['name'] == 'Chord'
        assert beat_seq[8][0]['name'] == 'Chord'
        assert beat_seq[12][0]['name'] == 'Chord'
        for b in range(16):
            if b in beat_seq:
                new_events.append(create_event('Beat', b))
                new_events += beat_seq[b]
        final_events += new_events
        full_start_new.append(len(final_events))
        final_events += full_song_events
    ls_start_new.append(len(final_events))
    assert len(ls_start_new) == len(full_start_new) + 1
    ls_start = ls_start_new
    full_start = full_start_new
    functional_events = final_events

    ls_position = [(ls_start[j], full_start[j]) for j in range(len(full_start_new))]
    full_position = [(full_start[j], ls_start[j + 1]) for j in range(len(full_start_new))]
    # print((ls_position, full_position, functional_events))

    return ls_position, full_position, functional_events


def event2full(skyline_pos, midi_pos, events, keyname, relative_chord, relative_melody):
    functional_events = []
    positions = []

    # --- emotion --- #
    emotion_event = create_event('Emotion', None)
    functional_events.append(emotion_event)

    # --- key --- #
    if relative_chord:
        key_event = create_event('Key', keyname)
        functional_events.append(key_event)

    # translate chord label to scale degree according to key
    root2degree = {MAJOR_KEY[i]: str(i) for i in range(len(MAJOR_KEY))}
    if relative_chord:
        start = np.where(MAJOR_KEY == keyname.upper())[0][0]
        scale_range = np.concatenate([MAJOR_KEY[start:], MAJOR_KEY[:start]], axis=0)
        root2degree = {scale_range[i]: str(i) for i in range(len(scale_range))}

    # --- global tempo --- #
    if events[0]['name'] == 'Tempo':
        global_key_event = events[0]
        functional_events.append(global_key_event)

    for pos in midi_pos:
        midi_events = events[pos[0] + 1:pos[1]]
        # print(midi_events)
        positions.append((len(functional_events)))
        bar_events = []
        beat_seq = defaultdict(list)
        for evs in range(len(midi_events)):
            if midi_events[evs]['name'] == 'Bar':
                bar_events.append(midi_events[evs])
            elif midi_events[evs]['name'] == 'Beat':
                beat_evs = '{}_{}'.format(midi_events[evs]['name'], midi_events[evs]['value'])
            else:
                beat_seq[beat_evs].append(midi_events[evs])

        prev_tempo = global_key_event
        prev_chord = create_event('Chord', 'None_None')
        for b in [0, 4, 8, 12]:
            beat_evs = 'Beat_{}'.format(b)
            if beat_evs not in beat_seq:
                beat_seq[beat_evs].append(prev_tempo)
                beat_seq[beat_evs].append(prev_chord)
            else:
                # --- add tempo --- #
                if beat_seq[beat_evs][0]['name'] == 'Tempo':
                    if beat_seq[beat_evs][0]['value'] == 'Conti':
                        beat_seq[beat_evs][0]['value'] = prev_tempo['value']
                    prev_tempo = beat_seq[beat_evs][0]
                else:
                    beat_seq[beat_evs] = [prev_tempo] + beat_seq[beat_evs]

                # --- add chord --- #
                if len(beat_seq[beat_evs]) == 1:
                    beat_seq[beat_evs] = [beat_seq[beat_evs][0], prev_chord]
                elif beat_seq[beat_evs][1]['name'] != 'Chord':
                    beat_seq[beat_evs] = [beat_seq[beat_evs][0], prev_chord] + beat_seq[beat_evs][1:]
                else:
                    if beat_seq[beat_evs][1]['value'] == 'Conti_Conti':
                        beat_seq[beat_evs][1]['value'] = prev_chord['value']
                    prev_chord = beat_seq[beat_evs][1]

        for b in range(16):
            beat_evs = 'Beat_{}'.format(b)
            if beat_evs in beat_seq:
                # --- beat --- #
                bar_events.append(create_event('Beat', b))
                for evs in beat_seq[beat_evs]:
                    # --- tempo --- #
                    if evs['name'] == 'Tempo':
                        bar_events.append(evs)

                    # --- chord --- #
                    elif evs['name'] == 'Chord':
                        if evs['value'] == 'None_None':
                            bar_events.append(evs)
                        else:
                            root, quality = evs['value'].split('_')
                            root = root2degree[IDX_TO_KEY[int(root)]]
                            # --- relative chord --- #
                            if relative_chord and root != 'None':
                                if keyname in MAJOR_KEY:
                                    root = majorDegree2roman[int(root)]
                                else:
                                    root = minorDegree2roman[int(root)]
                            chord_evs = create_event('Chord', root + '_' + quality)
                            bar_events.append(chord_evs)

                    # --- melody note --- #
                    elif evs['name'] == 'Note_Pitch':
                        # --- relative melody --- #
                        if relative_melody:
                            pitch = evs['value']
                            octave, roman = pitch2degree(keyname, pitch)
                            bar_events.append(create_event('Note_Octave', octave))
                            bar_events.append(create_event('Note_Degree', roman))
                        else:
                            bar_events.append(evs)

                    # --- melody duration / velocity --- #
                    else:
                        bar_events.append(evs)

        functional_events.extend(bar_events)

    # --- eos --- #
    functional_events.append(create_event('EOS', None))

    assert len(skyline_pos) == len(positions)

    # --- check events --- #
    count = defaultdict(int)
    for evs in functional_events:
        if evs['name'] == 'Chord':
            count['Chord'] += 1
        elif evs['name'] == 'Tempo':
            count['Tempo'] += 1
        elif evs['name'] == 'Beat':
            count['{}_{}'.format(evs['name'], evs['value'])] += 1
    assert count['Tempo'] == count['Chord'] + 1
    assert count['Beat_0'] + count['Beat_4'] + count['Beat_8'] + count['Beat_12'] == count['Chord']

    return positions, functional_events


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
    required.add_argument('-e', '--event_type',
                          choices=['lead2full', 'full'],
                          help='[lead2full] stage-two for two-stage models, '
                               '[full] one-stage models', required=True)
    args = parser.parse_args()

    representation = args.representation
    if representation == 'absolute':
        transpose_to_C, relative_chord, relative_melody = False, False, False
    elif representation == 'functional':
        transpose_to_C, relative_chord, relative_melody = False, True, True

    old_dir = 'midi_data/pop1k7/pop1k7_leedsheet2midi'
    event_type = args.event_type
    if event_type == 'lead2full':
        new_dir = 'events/stage2/pop1k7_events/full_song_chorder_{}/events'.format(representation)
    elif event_type == 'full':
        new_dir = 'events/stage1/pop1k7_events/full_song_chorder_{}/events'.format(representation)
    os.makedirs(new_dir, exist_ok=True)
    print(new_dir)

    # read keyname of midi files
    midi2key = json_read('midi_data/pop1k7/pop1k7_keyname.json')

    samples = os.listdir(old_dir)
    for i in tqdm(range(len(samples))):
        sample_name = samples[i]
        keyname = midi2key[sample_name[:-4]]
        f = os.path.join(old_dir, sample_name)
        # print(pickle.load(open(f, 'rb')))
        skyline_pos, midi_pos, events = pickle.load(open(f, 'rb'))

        if event_type == 'lead2full':
            ls_position, full_position, functional_events = event2lead_full(events=events, keyname=keyname,
                                                                            relative_chord=relative_chord,
                                                                            relative_melody=relative_melody)
            pickle_dump((ls_position, full_position, functional_events), os.path.join(new_dir, sample_name))
        elif event_type == 'full':
            positions, functional_events = event2full(skyline_pos=skyline_pos, midi_pos=midi_pos,
                                                      events=events, keyname=keyname,
                                                      relative_chord=relative_chord,
                                                      relative_melody=relative_melody)
            pickle_dump((positions, functional_events), os.path.join(new_dir, sample_name))
