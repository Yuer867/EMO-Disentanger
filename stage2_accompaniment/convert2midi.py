import miditoolkit
from miditoolkit.midi import containers
import numpy as np

##############################
# constants
##############################
DEFAULT_BEAT_RESOL = 480
DEFAULT_BAR_RESOL = 480 * 4
DEFAULT_FRACTION = 16

DEFAULT_SCALE = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
MAJOR_KEY = np.array(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
MINOR_KEY = np.array(['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b'])

quality_conversion_table = {
    #           1     2     3     4  5     6     7
    'maj':     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'min':     [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    'aug':     [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    'dim':     [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    'sus4':    [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    'sus4(b7)':[1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    'sus4(b7,9)':[1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    'sus2':    [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    '7':       [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj7':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min7':    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    'minmaj7': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    'maj6':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    'min6':    [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    '9':       [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj9':    [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min9':    [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    '7(#9)':   [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj6(9)': [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    'min6(9)': [1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    'maj(9)':  [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'min(9)':  [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    'maj(11)': [1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1],
    'min(11)': [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
    '11':      [1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0],
    'maj9(11)':[1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1],
    'min11':   [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
    '13':      [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
    'maj13':   [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    'min13':   [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
    'dim7':    [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    'hdim7':   [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    #'5':       [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    }
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
quality_name_table = {
    'M': 'maj',
    'm': 'min',
    '+': 'aug',
    'o': 'dim',
    'sus4': 'sus4',
    'sus2': 'sus2',
    '7': '7',
    'M7': 'maj7',
    'm7': 'min7',
    'o7': 'dim7',
    '/o7': 'hdim7',
    'None': 'None'
}


##############################
# containers for conversion
##############################
class ConversionEvent(object):
    def __init__(self, event, is_full_event=False):
        if not is_full_event:
            if 'Note' in event:
                self.name, self.value = '_'.join(event.split('_')[:-1]), event.split('_')[-1]
            elif 'Chord' in event:
                self.name, self.value = event.split('_')[0], '_'.join(event.split('_')[1:])
            else:
                self.name, self.value = event.split('_')
        else:
            self.name, self.value = event['name'], event['value']

    def __repr__(self):
        return 'Event(name: {} | value: {})'.format(self.name, self.value)


class NoteEvent(object):
    def __init__(self, pitch, bar, position, duration, velocity, microtiming=None):
        self.pitch = pitch
        self.start_tick = bar * DEFAULT_BAR_RESOL + position * (DEFAULT_BAR_RESOL // DEFAULT_FRACTION)
        self.duration = duration
        self.velocity = velocity

        if microtiming is not None:
            self.start_tick += microtiming

    def set_microtiming(self, microtiming):
        self.start_tick += microtiming

    def set_velocity(self, velocity):
        self.velocity = velocity

    def __repr__(self):
        return 'Note(pitch = {}, duration = {}, start_tick = {})'.format(
            self.pitch, self.duration, self.start_tick
        )


class TempoEvent(object):
    def __init__(self, tempo, bar, position):
        self.tempo = tempo
        self.start_tick = bar * DEFAULT_BAR_RESOL + position * (DEFAULT_BAR_RESOL // DEFAULT_FRACTION)

    def set_tempo(self, tempo):
        self.tempo = tempo

    def __repr__(self):
        return 'Tempo(tempo = {}, start_tick = {})'.format(
            self.tempo, self.start_tick
        )


class ChordEvent(object):
    def __init__(self, chord_val, bar, position):
        self.chord_val = chord_val
        self.start_tick = bar * DEFAULT_BAR_RESOL + position * (DEFAULT_BAR_RESOL // DEFAULT_FRACTION)


##############################
# conversion functions
##############################
def event_to_midi(key, events, mode, output_midi_path=None, is_full_event=False,
                  return_tempos=False, enforce_tempo=False, enforce_tempo_evs=None, play_chords=False):
    events = [ConversionEvent(ev, is_full_event=is_full_event) for ev in events]
    # print (events[:20])

    keyname = key.split('_')[1].upper()
    start = np.where(MAJOR_KEY == keyname)[0][0]
    scale_range = np.concatenate([MAJOR_KEY[start:], MAJOR_KEY[:start]], axis=0)

    # assert events[0].name == 'Bar'
    temp_notes = []
    temp_tempos = []
    temp_chords = []

    cur_bar = -1
    cur_position = 0

    for i in range(len(events)):
        if events[i].name == 'Bar':
            cur_bar += 1
        elif events[i].name == 'Beat':
            cur_position = int(events[i].value)
            assert cur_position >= 0 and cur_position < DEFAULT_FRACTION
        #   print (cur_bar, cur_position)
        elif events[i].name == 'Tempo' and 'Conti' not in events[i].value:
            temp_tempos.append(TempoEvent(
                int(events[i].value), max(cur_bar, 0), cur_position
            ))
        elif 'Note_Pitch' in events[i].name:
            if mode == 'full' and \
                    (i + 1) < len(events) and 'Note_Duration' in events[i + 1].name and \
                    (i + 2) < len(events) and 'Note_Velocity' in events[i + 2].name:
                # check if the 3 events are of the same instrument
                temp_notes.append(
                    NoteEvent(
                        pitch=int(events[i].value),
                        bar=cur_bar, position=cur_position,
                        duration=int(events[i + 1].value), velocity=int(events[i + 2].value)
                    )
                )
            elif mode == 'skyline' and \
                    (i + 1) < len(events) and 'Note_Duration' in events[i + 1].name:
                temp_notes.append(
                    NoteEvent(
                        pitch=int(events[i].value),
                        bar=cur_bar, position=cur_position,
                        duration=int(events[i + 1].value), velocity=80
                    )
                )
        elif 'Chord' in events[i].name and 'Conti' not in events[i].value:
            temp_chords.append(
                ChordEvent(events[i].value, cur_bar, cur_position)
            )
        elif events[i].name in ['EOS', 'PAD']:
            continue

    print('# tempo changes:', len(temp_tempos), '| # notes:', len(temp_notes))
    midi_obj = miditoolkit.midi.parser.MidiFile()
    midi_obj.instruments = [
        miditoolkit.Instrument(program=0, is_drum=False, name='Piano')
    ]

    for n in temp_notes:
        midi_obj.instruments[0].notes.append(
            miditoolkit.Note(int(n.velocity), n.pitch, int(n.start_tick), int(n.start_tick + n.duration))
        )

    if enforce_tempo is False:
        for t in temp_tempos:
            midi_obj.tempo_changes.append(
                miditoolkit.TempoChange(t.tempo, int(t.start_tick))
            )
    else:
        if enforce_tempo_evs is None:
            enforce_tempo_evs = temp_tempos[1]
        for t in enforce_tempo_evs:
            midi_obj.tempo_changes.append(
                miditoolkit.TempoChange(t.tempo, int(t.start_tick))
            )

    for c in temp_chords:
        if 'None' in c.chord_val:
            midi_obj.markers.append(
                miditoolkit.Marker('Chord-{}'.format(c.chord_val), int(c.start_tick))
            )
        else:
            chord_val = c.chord_val
            root = chord_val.split('_')[0]
            quality = chord_val.split('_')[1]
            c.chord_val = scale_range[int(root)] + '_' + quality
            midi_obj.markers.append(
                miditoolkit.Marker('Chord-{}'.format(c.chord_val), int(c.start_tick))
            )
    for b in range(cur_bar):
        midi_obj.markers.append(
            miditoolkit.Marker('Bar-{}'.format(b + 1), int(DEFAULT_BAR_RESOL * b))
        )

    midi_obj.max_tick = max([note.end for note in midi_obj.instruments[0].notes])

    if play_chords:
        add_chords(midi_obj)

    if output_midi_path is not None:
        midi_obj.dump(output_midi_path)

    if not return_tempos:
        return midi_obj
    else:
        return midi_obj, temp_tempos


def add_chords(midi_obj):
    default_velocity = 63

    markers = [marker for marker in midi_obj.markers if 'Chord' in marker.text]
    prev_chord = None
    dedup_chords = []
    for m in markers:
        if m.text == 'Chord-None_None':
            continue
        if m.text != prev_chord:
            prev_chord = m.text
            dedup_chords.append(m)
    markers = dedup_chords

    midi_maps = [chord_to_midi(marker.text.split('-')[1]) for marker in markers]
    midi_obj.instruments.append(containers.Instrument(program=0, is_drum=False, name='Piano'))

    if not len(midi_maps) == 0:
        for midi_map, prev_marker, next_marker in zip(midi_maps, markers[:-1], markers[1:]):
            for midi_pitch in midi_map:
                midi_note = containers.Note(start=prev_marker.time, end=next_marker.time, pitch=midi_pitch,
                                            velocity=default_velocity)
                midi_obj.instruments[1].notes.append(midi_note)
        for midi_pitch in midi_maps[-1]:
            midi_note = containers.Note(start=markers[-1].time, end=midi_obj.max_tick, pitch=midi_pitch,
                                        velocity=default_velocity)
            midi_obj.instruments[1].notes.append(midi_note)

    return midi_obj


def chord_to_midi(chord):
    root, quality = chord.split('_')
    bass = root

    root_c = 60
    bass_c = 36
    root_pc = KEY_TO_IDX[root]
    if quality in quality_name_table:
        quality = quality_name_table[quality]
    chord_map = list(np.where(np.array(quality_conversion_table[quality]) == 1)[0])
    bass_pc = KEY_TO_IDX[bass]

    return [bass_c + bass_pc] + [root_c + root_pc + i for i in chord_map]