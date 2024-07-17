import os
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import collections

import miditoolkit
from miditoolkit.midi import containers

from convert_key import pitch2degree, minorDegree2roman, majorDegree2roman

# ================================================== #
#  Configuration                                     #
# ================================================== #  
BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4
INSTR_NAME_MAP = {'piano': 0}
MIN_VELOCITY = 40
NOTE_SORTING = 1  # 0: ascending / 1: descending

DEFAULT_TEMPO = 110
DEFAULT_VELOCITY_BINS = np.linspace(4, 127, 42, dtype=int)
DEFAULT_BPM_BINS = np.linspace(32, 224, 64 + 1, dtype=int)
DEFAULT_SHIFT_BINS = np.linspace(-60, 60, 60 + 1, dtype=int)
DEFAULT_DURATION_BINS = np.arange(BEAT_RESOL / 8, BEAT_RESOL * 8 + 1, BEAT_RESOL / 8)
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

simplify_quality = {
    'maj': 'M',
    'min': 'm',
    'aug': '+',
    'dim': 'o',
    'sus4': 'sus4',
    'sus2': 'sus2',
    '7': '7',
    'maj7': 'M7',
    'min7': 'm7',
    'dim7': 'o7',
    'hdim7': '/o7',
    'None': 'None'
}


def analyzer(midi_path, keyname, only_melody=True, chord_conti=False, tempo_conti=False, relative_chord=False, transpose_to_C=False):
    """
    get melody and chord(marker) tracks for lead sheet
    """
    # load midi obj
    midi_obj = miditoolkit.midi.parser.MidiFile(midi_path)
    if only_melody:  # lead sheet
        notes = midi_obj.instruments[0].notes
        max_tick = max(note.end for note in notes)
    else:  # full song
        melody = midi_obj.instruments[0].notes
        texture = midi_obj.instruments[1].notes
        bass = midi_obj.instruments[2].notes
        notes = melody + texture + bass
        max_tick = midi_obj.max_tick
    notes = sorted(notes, key=lambda x: (x.start, x.pitch))

    # new midi obj
    new_midi_obj = miditoolkit.midi.parser.MidiFile()
    new_midi_obj.time_signature_changes.append(containers.TimeSignature(numerator=4, denominator=4, time=0))
    new_midi_obj.tempo_changes.append(containers.TempoChange(tempo=float(DEFAULT_TEMPO), time=0))
    new_midi_obj.instruments.append(containers.Instrument(program=0, is_drum=False, name='piano'))
    new_midi_obj.ticks_per_beat = BEAT_RESOL

    # --- melody --- #
    # remove overlap
    if only_melody:
        for i in range(len(notes[:-1])):
            notes[i].end = min(notes[i+1].start, notes[i].end)

    # whether transpose to C major / c minor
    if transpose_to_C:
        if KEY_TO_IDX[keyname.upper()] >= 6:
            pitch_offset = 12 - KEY_TO_IDX[keyname.upper()]
        else:
            pitch_offset = - KEY_TO_IDX[keyname.upper()]
        for note in notes:
            note.pitch = note.pitch + pitch_offset
            if note.pitch < 21 or note.pitch > 108:
                note.pitch = min(max(note.pitch, 21), 108)
    else:
        pitch_offset = 0

    new_midi_obj.instruments[0].notes = notes

    # --- chord --- #
    markers = midi_obj.markers

    # quantize and repeat / conti
    new_markers = []
    beat2chord = {}
    prev_chord = 'None_None_None'
    for chord in markers:
        quant_time = int(np.round(chord.time / BEAT_RESOL) * BEAT_RESOL)
        root, quality, bass = chord.text.split('_')
        beat2chord[quant_time] = root + '_' + simplify_quality[quality] + '_' + bass
    max_beat = int(np.ceil(max_tick / BEAT_RESOL) * BEAT_RESOL)
    for beat in range(0, max_beat, BEAT_RESOL):
        if beat in beat2chord:
            new_markers.append(containers.Marker(time=beat, text=beat2chord[beat]))
            prev_chord = beat2chord[beat]
        else:
            if chord_conti:
                new_markers.append(containers.Marker(time=beat, text='Conti_Conti_Conti'))
            else:
                new_markers.append(containers.Marker(time=beat, text=prev_chord))
    markers = new_markers

    # translate chord label to scale degree according to key
    root2degree = {MAJOR_KEY[i]: str(i) for i in range(len(MAJOR_KEY))}
    if relative_chord:
        start = np.where(MAJOR_KEY == keyname.upper())[0][0]
        scale_range = np.concatenate([MAJOR_KEY[start:], MAJOR_KEY[:start]], axis=0)
        root2degree = {scale_range[i]: str(i) for i in range(len(scale_range))}

    new_markers = []
    for m in markers:
        if 'None' in m.text or 'Conti' in m.text:
            new_markers.append(m)
        else:
            if transpose_to_C:
                root = IDX_TO_KEY[(KEY_TO_IDX[m.text.split('_')[0]] + pitch_offset) % 12]
                bass = IDX_TO_KEY[(KEY_TO_IDX[m.text.split('_')[2]] + pitch_offset) % 12]
            else:
                root = m.text.split('_')[0]
                bass = m.text.split('_')[2]
            root = root2degree[root]
            quality = m.text.split('_')[1]
            bass = root2degree[bass]
            text = '_'.join([root, quality, bass])
            new_markers.append(containers.Marker(time=m.time, text=text))
    assert len(new_markers) == len(markers)
    markers = new_markers

    new_midi_obj.markers = markers

    # --- global tempo --- #
    tempos = [b.tempo for b in midi_obj.tempo_changes][:40]
    tempo_median = np.median(tempos)
    global_bpm = int(tempo_median)
    new_midi_obj.markers.insert(0, containers.Marker(text='global_bpm_' + str(int(global_bpm)), time=0))

    # --- tempo --- #
    tempo_changes = []
    tick2tempo = {b.time: b for b in midi_obj.tempo_changes}
    prev_tempo = global_bpm
    for tick in range(0, (midi_obj.max_tick // BEAT_RESOL + 1) * BEAT_RESOL, BEAT_RESOL):
        if tick in tick2tempo:
            tempo_changes.append(tick2tempo[tick])
            prev_tempo = tick2tempo[tick]
        else:
            if tempo_conti:
                tempo_changes.append(containers.TempoChange(tempo='Conti', time=tick))
            else:
                tempo_changes.append(containers.TempoChange(tempo=prev_tempo.tempo, time=tick))
    new_midi_obj.tempo_changes = tempo_changes

    # --- key --- #
    if transpose_to_C:
        keyname = 'C' if keyname in MAJOR_KEY else 'c'
    new_midi_obj.markers.insert(0, containers.Marker(text='global_key_' + keyname, time=0))

    # save
    new_midi_obj.instruments[0].name = 'piano'
    return new_midi_obj


def midi2corpus(midi_obj):
    """
    quantize midi data
    """
    # load notes
    instr_notes = collections.defaultdict(list)
    for instr in midi_obj.instruments:
        # skip 
        if instr.name not in INSTR_NAME_MAP.keys():
            continue

        # process
        instr_idx = INSTR_NAME_MAP[instr.name]
        for note in instr.notes:
            note.instr_idx = instr_idx
            instr_notes[instr_idx].append(note)
        if NOTE_SORTING == 0:
            instr_notes[instr_idx].sort(
                key=lambda x: (x.start, x.pitch))
        elif NOTE_SORTING == 1:
            instr_notes[instr_idx].sort(
                key=lambda x: (x.start, -x.pitch))
        else:
            raise ValueError(' [x] Unknown type of sorting.')

    # load chords
    chords = []
    for marker in midi_obj.markers:
        if marker.text.split('_')[0] != 'global' and \
                'Boundary' not in marker.text.split('_')[0]:
            chords.append(marker)
    chords.sort(key=lambda x: x.time)

    # load tempos
    tempos = midi_obj.tempo_changes
    tempos.sort(key=lambda x: x.time)

    # load labels
    labels = []
    for marker in midi_obj.markers:
        if 'Boundary' in marker.text.split('_')[0]:
            labels.append(marker)
    labels.sort(key=lambda x: x.time)

    # load global bpm
    global_bpm = 120
    for marker in midi_obj.markers:
        if marker.text.split('_')[0] == 'global' and \
                marker.text.split('_')[1] == 'bpm':
            global_bpm = int(marker.text.split('_')[2])

    # load global key
    global_key = 'C'
    for marker in midi_obj.markers:
        if marker.text.split('_')[0] == 'global' and \
                marker.text.split('_')[1] == 'key':
            global_key = marker.text.split('_')[2]

    # --- process items to grid --- #
    # compute empty bar offset at head
    first_note_time = min([instr_notes[k][0].start for k in instr_notes.keys()])
    last_note_time = max([instr_notes[k][-1].start for k in instr_notes.keys()])

    quant_time_first = int(np.round(first_note_time / TICK_RESOL) * TICK_RESOL)
    offset = quant_time_first // BAR_RESOL  # empty bar
    last_bar = int(np.ceil(last_note_time / BAR_RESOL)) - offset
    # print(' > offset:', offset)
    # print(' > last_bar:', last_bar)

    # process notes
    instr_gird = dict()
    for key in instr_notes.keys():
        notes = instr_notes[key]
        note_grid = collections.defaultdict(list)
        for note in notes:
            note.start = note.start - offset * BAR_RESOL
            note.end = note.end - offset * BAR_RESOL

            # quantize start
            quant_time = int(np.round(note.start / TICK_RESOL) * TICK_RESOL)

            # velocity
            note.velocity = DEFAULT_VELOCITY_BINS[
                np.argmin(abs(DEFAULT_VELOCITY_BINS - note.velocity))]
            # note.velocity = max(MIN_VELOCITY, note.velocity)

            # shift of start
            note.shift = note.start - quant_time
            note.shift = DEFAULT_SHIFT_BINS[np.argmin(abs(DEFAULT_SHIFT_BINS - note.shift))]

            # duration
            note_duration = note.end - note.start
            if note_duration > BAR_RESOL:
                note_duration = BAR_RESOL
            ntick_duration = int(np.round(note_duration / TICK_RESOL) * TICK_RESOL)
            if ntick_duration == 0:
                continue
            note.duration = ntick_duration

            # append
            note_grid[quant_time].append(note)

        # set to track
        instr_gird[key] = note_grid.copy()

    # process chords
    chord_grid = collections.defaultdict(list)
    for chord in chords:
        # quantize
        chord.time = chord.time - offset * BAR_RESOL
        chord.time = 0 if chord.time < 0 else chord.time
        quant_time = int(np.round(chord.time / TICK_RESOL) * TICK_RESOL)

        # append
        chord_grid[quant_time].append(chord)

    # remove multiple chords in one grid
    for q in chord_grid:
        if len(chord_grid[q]) > 1:
            for c in chord_grid[q][::-1]:
                if not c.text == 'Conti_Conti_Conti':
                    chord_grid[q] = [c]
                    break

    # process tempo
    tempo_grid = collections.defaultdict(list)
    for tempo in tempos:
        # quantize
        tempo.time = tempo.time - offset * BAR_RESOL
        tempo.time = 0 if tempo.time < 0 else tempo.time
        quant_time = int(np.round(tempo.time / TICK_RESOL) * TICK_RESOL)
        if tempo.tempo != 'Conti':
            tempo.tempo = DEFAULT_BPM_BINS[np.argmin(abs(DEFAULT_BPM_BINS - tempo.tempo))]

        # append
        tempo_grid[quant_time].append(tempo)

    # remove multiple tempos in one grid
    for q in tempo_grid:
        if len(tempo_grid[q]) > 1:
            for t in tempo_grid[q][::-1]:
                if not t.tempo == 'Conti':
                    tempo_grid[q] = [t]
                    break

    # process boundary
    label_grid = collections.defaultdict(list)
    for label in labels:
        # quantize
        label.time = label.time - offset * BAR_RESOL
        label.time = 0 if label.time < 0 else label.time
        quant_time = int(np.round(label.time / TICK_RESOL) * TICK_RESOL)

        # append
        label_grid[quant_time] = [label]

    # process global bpm
    global_bpm = DEFAULT_BPM_BINS[np.argmin(abs(DEFAULT_BPM_BINS - global_bpm))]

    # collect
    song_data = {
        'notes': instr_gird,
        'chords': chord_grid,
        'tempos': tempo_grid,
        'labels': label_grid,
        'metadata': {
            'global_bpm': global_bpm,
            'last_bar': last_bar,
            'global_key': global_key
        }
    }

    return song_data


def create_event(name, value):
    event = dict()
    event['name'] = name
    event['value'] = value
    return event


def corpus2lead(data, emotion=None, relative_melody=False, relative_chord=False):
    """
        convert data to lead sheet events
        (1) relative = False
        - Emotion - (Key) - Track(Melody) - Bar - Beat_0 - Note_Pitch - Note_Duration - Beat_1 - ... - EOS
        - Track(Chord) - Bar - Beat_0 - Chord_0_M - Beat_4 - Chord_0_M - Beat_8 - ...
        (2) relative = True
        - Emotion - (Key) - Track(Melody) - Bar - Beat_0 - Note_Octave - Note_Degree - Note_Duration - Beat_1 - ... - EOS
        - Track(Chord) - Bar - Beat_0 - Chord_I_M - Beat_4 - Chord_I_M - Beat_8 - ...
        """
    # global tag
    global_end = data['metadata']['last_bar'] * BAR_RESOL

    # process
    position = []
    final_sequence = []

    # --- tempo --- #
    # global_bpm = data['metadata']['global_bpm']
    # final_sequence.append(create_event('Tempo', global_bpm))

    # --- emotion --- #
    final_sequence.append(create_event('Emotion', emotion))

    # --- key --- #
    global_key = data['metadata']['global_key']
    if relative_chord:
        final_sequence.append(create_event('Key', global_key))

    for bar_step in range(0, global_end, BAR_RESOL):
        sequence = [create_event('Bar', None)]

        # --- piano track --- #
        for timing in range(bar_step, bar_step + BAR_RESOL, TICK_RESOL):
            events = []

            # unpack
            t_chords = data['chords'][timing]
            t_tempos = data['tempos'][timing]
            t_notes = data['notes'][0][timing]

            # chord
            if len(t_chords):
                root, quality, bass = t_chords[0].text.split('_')
                if relative_melody and root != 'None':
                    if global_key in MAJOR_KEY:
                        root = majorDegree2roman[int(root)]
                    else:
                        root = minorDegree2roman[int(root)]
                events.append(create_event('Chord', root + '_' + quality))

            # # tempo
            # if len(t_tempos) and tempo == 'time-varying':
            #     melody_events.append(create_event('Tempo', t_tempos[0].key))

            # note
            if len(t_notes):
                for note in t_notes:
                    if relative_melody:
                        octave, roman = pitch2degree(global_key, note.pitch)
                        events.extend([
                            create_event('Note_Octave', octave),
                            create_event('Note_Degree', roman),
                            create_event('Note_Duration', note.duration),
                            # create_event('Note_Velocity', note.velocity),
                        ])
                    else:
                        events.extend([
                            create_event('Note_Pitch', note.pitch),
                            create_event('Note_Duration', note.duration),
                            # create_event('Note_Velocity', note.velocity),
                        ])

            # collect & beat
            if len(events):
                sequence.append(
                    create_event('Beat', (timing - bar_step) // TICK_RESOL))
                sequence.extend(events)

        # --- EOS --- #
        if bar_step == global_end - BAR_RESOL:
            sequence.append(create_event('EOS', None))

        # --- align two tracks --- #
        position.append(len(final_sequence))
        final_sequence.extend(sequence)

    return position, final_sequence


def corpus2full(lead_data, full_data, emotion=None, relative_melody=False, relative_chord=False):
    """
        convert data to full song events
        e.g.
        - Tempo(global) - Track(Lead) - Bar - Emotion -  Beat_0 - Chord - Note_Pitch - Note_Duration - Beat_1 - ... - EOS
        - Track(Full) - Bar - Beat_0 - Tempo(time-varying) - Chord - Note_Pitch - Note_Duration - Note_Velocity - ...
        """
    # global tag
    global_end = lead_data['metadata']['last_bar'] * BAR_RESOL

    # process
    lead_position = []
    full_position = []
    final_sequence = []

    # --- emotion --- #
    final_sequence.append(create_event('Emotion', emotion))

    # --- key --- #
    global_key = lead_data['metadata']['global_key']
    if relative_chord:
        final_sequence.append(create_event('Key', global_key))

    # --- global tempo --- #
    global_bpm = lead_data['metadata']['global_bpm']
    final_sequence.append(create_event('Tempo', global_bpm))

    for bar_step in range(0, global_end, BAR_RESOL):
        lead_sequence = [create_event('Track', 'LeadSheet'), create_event('Bar', None)]
        full_sequence = [create_event('Track', 'Full'), create_event('Bar', None)]

        # --- piano track --- #
        for timing in range(bar_step, bar_step + BAR_RESOL, TICK_RESOL):
            # --- lead sheet --- #
            lead_events = []

            # chord
            t_chords = lead_data['chords'][timing]
            if len(t_chords):
                root, quality, bass = t_chords[0].text.split('_')
                if relative_melody and root != 'None':
                    if global_key in MAJOR_KEY:
                        root = majorDegree2roman[int(root)]
                    else:
                        root = minorDegree2roman[int(root)]
                lead_events.append(create_event('Chord', root + '_' + quality))

            # note
            t_notes = lead_data['notes'][0][timing]
            if len(t_notes):
                for note in t_notes:
                    if relative_melody:
                        octave, roman = pitch2degree(global_key, note.pitch)
                        lead_events.extend([
                            create_event('Note_Octave', octave),
                            create_event('Note_Degree', roman),
                            # create_event('Note_Velocity', note.velocity),
                            create_event('Note_Duration', note.duration),
                        ])
                    else:
                        lead_events.extend([
                            create_event('Note_Pitch', note.pitch),
                            # create_event('Note_Velocity', note.velocity),
                            create_event('Note_Duration', note.duration),
                        ])

            # collect & beat
            if len(lead_events):
                lead_sequence.append(
                    create_event('Beat', (timing - bar_step) // TICK_RESOL))
                lead_sequence.extend(lead_events)

            # --- full song --- #
            full_events = []

            # tempo
            t_tempos = full_data['tempos'][timing]
            if len(t_tempos):
                full_events.append(create_event('Tempo', t_tempos[0].tempo))

            # chord
            t_chords = full_data['chords'][timing]
            if len(t_chords):
                root, quality, bass = t_chords[0].text.split('_')
                if relative_melody and root not in ['None', 'Conti']:
                    if global_key in MAJOR_KEY:
                        root = majorDegree2roman[int(root)]
                    else:
                        root = minorDegree2roman[int(root)]
                full_events.append(create_event('Chord', root + '_' + quality))

            # note
            t_notes = full_data['notes'][0][timing]
            if len(t_notes):
                for note in t_notes:
                    if relative_melody:
                        octave, roman = pitch2degree(global_key, note.pitch)
                        full_events.extend([
                            create_event('Note_Octave', octave),
                            create_event('Note_Degree', roman),
                            create_event('Note_Duration', note.duration),
                            create_event('Note_Velocity', note.velocity),
                        ])
                    else:
                        full_events.extend([
                            create_event('Note_Pitch', note.pitch),
                            create_event('Note_Duration', note.duration),
                            create_event('Note_Velocity', note.velocity),
                        ])

            if len(full_events):
                full_sequence.append(
                    create_event('Beat', (timing - bar_step) // TICK_RESOL))
                full_sequence.extend(full_events)

        # --- EOS --- #
        if bar_step == global_end - BAR_RESOL:
            lead_sequence.append(create_event('EOS', None))

        # --- align two tracks --- #
        lead_start = len(final_sequence)
        final_sequence.extend(lead_sequence)
        lead_end = len(final_sequence)
        lead_position.append((lead_start, lead_end))

        full_start = len(final_sequence)
        final_sequence.extend(full_sequence)
        full_end = len(final_sequence)
        full_position.append((full_start, full_end))

    return lead_position, full_position, final_sequence


def corpus2lead_full(data, emotion=None, relative_melody=False, relative_chord=False):
    """
        convert data to lead sheet events
        (1) relative = False
        - Emotion - (Key) - Track(Melody) - Bar - Beat_0 - Note_Pitch - Note_Duration - Beat_1 - ... - EOS
        - Track(Chord) - Bar - Beat_0 - Chord_0_M - Beat_4 - Chord_0_M - Beat_8 - ...
        (2) relative = True
        - Emotion - (Key) - Track(Melody) - Bar - Beat_0 - Note_Octave - Note_Degree - Note_Duration - Beat_1 - ... - EOS
        - Track(Chord) - Bar - Beat_0 - Chord_I_M - Beat_4 - Chord_I_M - Beat_8 - ...
        """
    # global tag
    global_end = data['metadata']['last_bar'] * BAR_RESOL

    # process
    position = []
    final_sequence = []

    # --- emotion --- #
    final_sequence.append(create_event('Emotion', emotion))

    # --- key --- #
    global_key = data['metadata']['global_key']
    if relative_chord:
        final_sequence.append(create_event('Key', global_key))
    # final_sequence.append(create_event('Key', global_key))

    # --- tempo --- #
    global_bpm = data['metadata']['global_bpm']
    final_sequence.append(create_event('Tempo', global_bpm))

    for bar_step in range(0, global_end, BAR_RESOL):
        sequence = [create_event('Bar', None)]

        # --- piano track --- #
        for timing in range(bar_step, bar_step + BAR_RESOL, TICK_RESOL):
            events = []

            # unpack
            t_chords = data['chords'][timing]
            t_tempos = data['tempos'][timing]
            t_notes = data['notes'][0][timing]

            # tempo
            if len(t_tempos):
                events.append(create_event('Tempo', t_tempos[0].tempo))

            # chord
            if len(t_chords):
                root, quality, bass = t_chords[0].text.split('_')
                if relative_melody and root != 'None':
                    if global_key in MAJOR_KEY:
                        root = majorDegree2roman[int(root)]
                    else:
                        root = minorDegree2roman[int(root)]
                events.append(create_event('Chord', root + '_' + quality))

            # note
            if len(t_notes):
                for note in t_notes:
                    if relative_melody:
                        octave, roman = pitch2degree(global_key, note.pitch)
                        events.extend([
                            create_event('Note_Octave', octave),
                            create_event('Note_Degree', roman),
                            create_event('Note_Duration', note.duration),
                            create_event('Note_Velocity', note.velocity),
                        ])
                    else:
                        events.extend([
                            create_event('Note_Pitch', note.pitch),
                            create_event('Note_Duration', note.duration),
                            create_event('Note_Velocity', note.velocity),
                        ])

            # collect & beat
            if len(events):
                sequence.append(
                    create_event('Beat', (timing - bar_step) // TICK_RESOL))
                sequence.extend(events)

        # --- EOS --- #
        if bar_step == global_end - BAR_RESOL:
            sequence.append(create_event('EOS', None))

        # --- align two tracks --- #
        position.append(len(final_sequence))
        final_sequence.extend(sequence)

    return position, final_sequence


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
                          choices=['lead', 'lead2full', 'full'],
                          help='[lead] stage-one for two-stage models, '
                               '[lead2full] stage-two for two-stage models, '
                               '[full] one-stage models', required=True)
    args = parser.parse_args()

    representation = args.representation
    if representation == 'absolute':
        transpose_to_C, relative_chord, relative_melody = False, False, False
    elif representation == 'functional':
        transpose_to_C, relative_chord, relative_melody = False, True, True

    event_type = args.event_type
    if event_type == 'lead':
        num_emotion = 2
        emopia_data_home = 'midi_data/EMOPIA/extracted_midis_480_chord11_tempo_clean'
        lead_sheet_events_dir = '../stage1_compose/emopia_events/lead_sheet_chord11_{}_adjusted_clean/events'.format(
            representation)
        os.makedirs(lead_sheet_events_dir, exist_ok=True)
        print('save dir:', lead_sheet_events_dir)
    elif event_type == 'lead2full':
        num_emotion = 4
        emopia_data_home = 'midi_data/EMOPIA/extracted_midis_480_chord11_tempo_clean'
        full_song_events_dir = '../stage2_accompaniment/emopia_events/full_song_chord11_{}_adjusted_clean/events'.format(
            representation)
        os.makedirs(full_song_events_dir, exist_ok=True)
        print('save dir:', full_song_events_dir)
    elif event_type == 'full':
        num_emotion = 4
        emopia_data_home = 'midi_data/EMOPIA/extracted_midis_480_chord11_tempo'
        full_song_events_dir = '../stage1_compose/emopia_events/full_song_chord11_{}_adjusted/events'.format(
            representation)
        os.makedirs(full_song_events_dir, exist_ok=True)
        print('save dir:', full_song_events_dir)
    midi_files = os.listdir(emopia_data_home)

    # load dict for key
    # clip2keyname, _ = find_key_emopia()
    with open('midi_data/EMOPIA/adjust_keyname.json', 'r') as f:
        clip2keyname = json.load(f)

    print('convert midi to events ...')
    for file in tqdm(midi_files):
        filename = file[:-4]
        midi_path = os.path.join(emopia_data_home, filename + '.mid')

        # get key tag
        keyname = clip2keyname[filename]

        # get emotion tag (only consider High/Low Valence)
        emotion = filename[:2]
        if num_emotion == 2:  # only consider High/Low Valence
            if emotion in ['Q1', 'Q4']:
                emotion = 'Positive'  # High Valence
            elif emotion in ['Q2', 'Q3']:
                emotion = 'Negative'  # Low Valence

        # === convert midi to lead sheet === #
        if event_type == 'lead':
            lead_midi_obj = analyzer(midi_path, keyname, only_melody=True, chord_conti=False, tempo_conti=False,
                                     relative_chord=relative_chord, transpose_to_C=transpose_to_C)
            lead_data = midi2corpus(lead_midi_obj)
            pos, lead_events = corpus2lead(lead_data, emotion,
                                           relative_melody=relative_melody, relative_chord=relative_chord)
            pickle.dump((pos, lead_events),
                        open(os.path.join(lead_sheet_events_dir, filename + '.pkl'), 'wb'))

        # === convert midi to lead sheet & full song === #
        if event_type == 'lead2full':
            lead_midi_obj = analyzer(midi_path, keyname, only_melody=True, chord_conti=False, tempo_conti=True,
                                     relative_chord=relative_chord, transpose_to_C=transpose_to_C)
            lead_data = midi2corpus(lead_midi_obj)

            full_midi_obj = analyzer(midi_path, keyname, only_melody=False, chord_conti=True, tempo_conti=True,
                                     relative_chord=relative_chord, transpose_to_C=transpose_to_C)
            full_data = midi2corpus(full_midi_obj)
            lead_pos, full_pos, full_events = corpus2full(lead_data, full_data, emotion,
                                                          relative_melody=relative_melody, relative_chord=relative_chord)
            pickle.dump((lead_pos, full_pos, full_events),
                        open(os.path.join(full_song_events_dir, filename + '.pkl'), 'wb'))

        # === convert midi to full song === #
        if event_type == 'full':
            full_midi_obj = analyzer(midi_path, keyname, only_melody=False, chord_conti=False, tempo_conti=False,
                                     relative_chord=relative_chord, transpose_to_C=transpose_to_C)
            full_data = midi2corpus(full_midi_obj)
            pos, full_events = corpus2lead_full(full_data, emotion,
                                                relative_melody=relative_melody, relative_chord=relative_chord)
            pickle.dump((pos, full_events),
                        open(os.path.join(full_song_events_dir, filename + '.pkl'), 'wb'))
