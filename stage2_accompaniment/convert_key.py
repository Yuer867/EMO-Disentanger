import os
import csv
import gzip
import json
import random
import numpy as np
import collections
from tqdm import tqdm


emopia_data_home = '../representations/midi_data/EMOPIA/'
hooktheory_data_home = '../representations/midi_data/HookTheory'

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
KEY_TO_IDX = {v: k for k, v in IDX_TO_KEY.items()}

majorDegree2roman = {
    0: 'I',
    1: 'I#',
    2: 'II',
    3: 'II#',
    4: 'III',
    5: 'IV',
    6: 'IV#',
    7: 'V',
    8: 'V#',
    9: 'VI',
    10: 'VI#',
    11: 'VII',
}
roman2majorDegree = {v: k for k, v in majorDegree2roman.items()}

minorDegree2roman = {
    0: 'I',
    1: 'I#',
    2: 'II',
    3: 'III',
    4: random.choice(['III', 'IV']),
    5: 'IV',
    6: 'IV#',
    7: 'V',
    8: 'VI',
    9: 'VI#',
    10: 'VII',
    11: random.choice(['VII', 'I'])
}
roman2minorDegree = {
    'I': 0,
    'I#': 1,
    'II': 2,
    'II#': random.choice([2, 3]),
    'III': 3,
    'IV': 5,
    'IV#': 6,
    'V': 7,
    'V#': random.choice([7, 8]),
    'VI': 8,
    'VI#': 9,
    'VII': 10
}


def find_key_emopia():
    print('load keyname for emopia clips ...')
    header, content = csv_read(os.path.join(emopia_data_home, 'key_mode_tempo.csv'))
    clip2keyname = collections.defaultdict(str)
    clip2keymode = collections.defaultdict(str)
    for c in content:
        name = c[1]
        keyname = c[2]
        keymode = 0 if keyname in MAJOR_KEY else 1
        clip2keyname[name] = keyname
        clip2keymode[name] = keymode
    return clip2keyname, clip2keymode


def find_key_hooktheory():
    print('load keyname for HookTheory clips ...')
    with gzip.open(os.path.join(hooktheory_data_home, 'Hooktheory.json.gz'), 'r') as f:
        dataset = json.load(f)

    clip2keyname = dict()
    clip2keymode = dict()
    for k, v in tqdm(dataset.items()):
        clip_name = k
        annotations = v['annotations']
        key = IDX_TO_KEY[annotations['keys'][0]['tonic_pitch_class']]
        mode = list2str(annotations['keys'][0]['scale_degree_intervals'])

        if mode == '221222':
            clip2keyname[clip_name] = key.upper()
            clip2keymode[clip_name] = 0
        elif mode == '212212':
            clip2keyname[clip_name] = key.lower()
            clip2keymode[clip_name] = 1
        else:
            continue

    return clip2keyname, clip2keymode


def pitch2degree(key, pitch):
    degree = pitch % 12

    # major key
    if key in MAJOR_KEY:
        tonic = KEY_TO_IDX[key]
        degree = (degree + 12 - tonic) % 12
        octave = (pitch - degree) // 12
        roman = majorDegree2roman[degree]
    # minor key
    elif key in MINOR_KEY:
        tonic = KEY_TO_IDX[key.upper()]
        degree = (degree + 12 - tonic) % 12
        octave = (pitch - degree) // 12
        roman = minorDegree2roman[degree]
    else:
        raise NameError('Wrong key name {}.'.format(key))

    return octave, roman


def degree2pitch(key, octave, roman):
    # major key
    if key in MAJOR_KEY:
        tonic = KEY_TO_IDX[key]
        pitch = octave * 12 + tonic + roman2majorDegree[roman]
    # minor key
    elif key in MINOR_KEY:
        tonic = KEY_TO_IDX[key.upper()]
        pitch = octave * 12 + tonic + roman2minorDegree[roman]
    else:
        raise NameError('Wrong key name {}.'.format(key))

    return pitch


def absolute2relative(events, enforce_key=False, enforce_key_evs=None):
    if enforce_key:
        key = enforce_key_evs['value']
    else:
        for evs in events:
            if evs['name'] == 'Key':
                key = evs['value']
                break

    new_events = []
    for evs in events:
        if evs['name'] == 'Key':
            new_events.append({'name': 'Key', 'value': key})
        elif evs['name'] == 'Note_Pitch':
            pitch = evs['value']
            octave, roman = pitch2degree(key, pitch)
            new_events.append({'name': 'Note_Octave', 'value': octave})
            new_events.append({'name': 'Note_Degree', 'value': roman})
        else:
            new_events.append(evs)

    return new_events


def relative2absolute(events, enforce_key=False, enforce_key_evs=None):
    if enforce_key:
        key = enforce_key_evs['value']
    else:
        for evs in events:
            if evs['name'] == 'Key':
                key = evs['value']
                break

    new_events = []
    for evs in events:
        if evs['name'] == 'Key':
            new_events.append({'name': 'Key', 'value': key})
        elif evs['name'] == 'Note_Octave':
            octave = evs['value']
        elif evs['name'] == 'Note_Degree':
            roman = evs['value']
            pitch = degree2pitch(key, octave, roman)
            pitch = max(21, pitch)
            pitch = min(108, pitch)
            if pitch < 21 or pitch > 108:
                raise ValueError('Pitch value must be in (21, 108), but gets {}'.format(pitch))
            new_events.append({'name': 'Note_Pitch', 'value': pitch})
        else:
            new_events.append(evs)

    return new_events


def switch_key(key):
    if '_' in key:
        keyname = key.split('_')[1]
        if keyname in MAJOR_KEY:
            return 'Key_' + keyname.lower()
        if keyname in MINOR_KEY:
            return 'Key_' + keyname.upper()
    if key in MAJOR_KEY:
        return key.lower()
    if key in MINOR_KEY:
        return key.upper()


def switch_melody(filename, events, clip2keymode):
    keymode = int(clip2keymode[filename])
    # if positive & minor / negative & major, not switch
    if (filename[:2] in ['Q1', 'Q4'] and keymode == 1) or (filename[:2] in ['Q2', 'Q3'] and keymode == 0):
        return events
    # if positive & major / negative & minor, switch from major to minor / minor to major
    else:
        keyname = 'C' if keymode == 0 else 'c'
        key_event = {'name': 'Key', 'value': keyname}
        new_events = absolute2relative(events, enforce_key=True, enforce_key_evs=key_event)

        new_key_event = {'name': 'Key', 'value': switch_key(keyname)}
        new_events = relative2absolute(new_events, enforce_key=True, enforce_key_evs=new_key_event)
        return new_events


def csv_read(path):
    content = list()
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            content.append(row)
    f.close()
    header = content[0]
    content = content[1:]
    return header, content


def list2str(a_list):
    return ''.join([str(i) for i in a_list])