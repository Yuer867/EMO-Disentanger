import os
import pickle
import argparse
import numpy as np

from convert_key import majorDegree2roman

BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4
DEFAULT_SCALE = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
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


def build_full_vocab(add_velocity=True, add_emotion=True, add_tempo=True,
                     num_emotion=4, relative=False):
    vocab = []

    # --- emotion --- #
    if add_emotion:
        if num_emotion == 2:
            for e in ['Positive', 'Negative', None]:
                vocab.append('Emotion_{}'.format(e))
        else:
            for e in ['Q1', 'Q2', 'Q3', 'Q4', None]:
                vocab.append('Emotion_{}'.format(e))

    # --- chord --- #
    # scale
    scale = [KEY_TO_IDX[s] for s in DEFAULT_SCALE]
    if relative:
        scale = [majorDegree2roman[s] for s in scale]

    # quality
    standard_qualities = ['M', 'm', 'o', '+', '7', 'M7', 'm7', 'o7', '/o7', 'sus2', 'sus4']
    print(' > num qualities:', len(standard_qualities))

    # combination
    for s in scale:
        for q in standard_qualities:
            vocab.append('Chord_{}_{}'.format(s, q))
    vocab.append('Chord_None_None')

    # --- note --- #
    if relative:
        # note octave
        for o in range(21//12, 109//12+1):
            vocab.append('Note_Octave_{}'.format(o))
        # note degree
        for d in list(majorDegree2roman.values()):
            vocab.append('Note_Degree_{}'.format(d))
    else:
        # note pitch
        for p in range(21, 109):
            vocab.append('Note_Pitch_{}'.format(p))
    # note velocity
    if add_velocity:
        for v in np.linspace(4, 127, 42, dtype=int):
            vocab.append('Note_Velocity_{}'.format(int(v)))
    # note duration
    for d in np.arange(TICK_RESOL, BAR_RESOL + TICK_RESOL, TICK_RESOL):
        vocab.append('Note_Duration_{}'.format(int(d)))

    # --- tempo --- #
    if add_tempo:
        for t in np.linspace(32, 224, 64 + 1, dtype=int):
            vocab.append('Tempo_{}'.format(int(t)))

    return vocab


def events2dictionary(root, add_velocity=False, add_emotion=True, add_tempo=True,
                      num_emotion=4, relative=False, event_pos=2):
    event_path = os.path.join(root, 'events')
    dictionary_path = os.path.join(root, 'dictionary.pkl')

    # list files
    event_files = os.listdir(event_path)
    n_files = len(event_files)
    print(' > num files:', n_files)

    # generate dictionary
    all_events = []
    for file in event_files:
        events = pickle.load(open(os.path.join(event_path, file), 'rb'))[event_pos]
        for event in events:
            all_events.append('{}_{}'.format(event['name'], event['value']))
    all_events = all_events + build_full_vocab(add_velocity=add_velocity,
                                               add_emotion=add_emotion,
                                               add_tempo=add_tempo,
                                               num_emotion=num_emotion,
                                               relative=relative)
    unique_events = sorted(set(all_events), key=lambda x: (not isinstance(x, int), x))
    event2word = {key: i for i, key in enumerate(unique_events)}
    word2event = {i: key for i, key in enumerate(unique_events)}
    # print(event2word)
    print(' > num classes:', len(word2event))
    print()
    # print(word2event)

    # save
    pickle.dump((event2word, word2event), open(dictionary_path, 'wb'))


if __name__ == '__main__':
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
        relative = False
    elif representation == 'functional':
        relative = True
    else:
        raise ValueError("invalid representation {}, choose from ['absolute', 'functional']".format(representation))

    # ======== stage-one for two-stage models ========#
    events_dir = 'events/stage1/hooktheory_events/lead_sheet_chord11_{}'.format(representation)
    print(events_dir)
    events2dictionary(events_dir, add_velocity=False, add_emotion=True, add_tempo=False,
                      num_emotion=2, relative=relative, event_pos=1)

    events_dir = 'events/stage1/emopia_events/lead_sheet_chord11_{}_adjusted_clean'.format(representation)
    print(events_dir)
    events2dictionary(events_dir, add_velocity=False, add_emotion=True, add_tempo=False,
                      num_emotion=2, relative=relative, event_pos=1)

    # ======== stage-two for two-stage models ========#
    events_dir = 'events/stage2/pop1k7_events/full_song_chorder_{}'.format(representation)
    print(events_dir)
    events2dictionary(events_dir, add_velocity=True, add_emotion=True, add_tempo=True,
                      num_emotion=4, relative=relative, event_pos=2)

    events_dir = 'events/stage2/emopia_events/full_song_chord11_{}_adjusted_clean'.format(representation)
    print(events_dir)
    events2dictionary(events_dir, add_velocity=True, add_emotion=True, add_tempo=True,
                      num_emotion=4, relative=relative, event_pos=2)

    # ======== one-stage models ========#
    events_dir = 'events/stage1/pop1k7_events/full_song_chorder_{}'.format(representation)
    print(events_dir)
    events2dictionary(events_dir, add_velocity=True, add_emotion=True, add_tempo=True,
                      num_emotion=4, relative=relative, event_pos=1)

    events_dir = 'events/stage1/emopia_events/full_song_chord11_{}_adjusted'.format(representation)
    print(events_dir)
    events2dictionary(events_dir, add_velocity=True, add_emotion=True, add_tempo=True,
                      num_emotion=4, relative=relative, event_pos=1)
