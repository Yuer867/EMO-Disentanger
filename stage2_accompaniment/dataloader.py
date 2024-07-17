import os
import random
import numpy as np
from glob import glob

import torch
from torch.utils.data import Dataset

from utils import pickle_load

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


def convert_event(event_seq, event2idx, to_ndarr=True):
    if isinstance(event_seq[0], dict):
        event_seq = [event2idx['{}_{}'.format(e['name'], e['value'])] for e in event_seq]
    else:
        event_seq = [event2idx[e] for e in event_seq]

    if to_ndarr:
        return np.array(event_seq)
    else:
        return event_seq


class REMISkylineToMidiTransformerDataset(Dataset):
    def __init__(self, data_dir, vocab_file,
                 model_dec_seqlen=10240, model_max_bars=None,
                 pieces=[], pad_to_same=True, appoint_st_bar=None,
                 dec_end_pad_value=None, predict_key=None
                 ):
        self.vocab_file = vocab_file
        self.read_vocab()

        self.model_dec_seqlen = model_dec_seqlen
        self.model_max_bars = model_max_bars
        self.data_dir = data_dir
        self.pieces = pieces
        self.build_dataset()

        self.pad_to_same = pad_to_same
        self.predict_key = predict_key

        self.appoint_st_bar = appoint_st_bar
        if dec_end_pad_value == 'EOS':
            self.dec_end_pad_value = self.eos_token
        else:
            self.dec_end_pad_value = self.pad_token

    def read_vocab(self):
        vocab = pickle_load(self.vocab_file)[0]
        self.idx2event = pickle_load(self.vocab_file)[1]
        orig_vocab_size = len(vocab)
        self.event2idx = vocab
        self.bar_token = self.event2idx['Bar_None']
        self.eos_token = self.event2idx['EOS_None']
        self.pad_token = orig_vocab_size
        self.vocab_size = self.pad_token + 1

    def build_dataset(self):
        if not self.pieces:
            self.pieces = sorted(glob(os.path.join(self.data_dir, '*.pkl')))
        else:
            self.pieces = sorted([os.path.join(self.data_dir, p) for p in self.pieces])

        self.piece_melody_pos = []
        self.piece_chord_pos = []
        self.piece_admissible_stbars = []

        for i, p in enumerate(self.pieces):
            piece_data = pickle_load(p)
            melody_pos, chord_pos = piece_data[0], piece_data[1]
            piece_evs = piece_data[2]
            if not i % 200:
                print('[preparing data] now at #{}'.format(i))

            self.piece_melody_pos.append(melody_pos)
            self.piece_chord_pos.append(chord_pos)

            if len(piece_evs) <= self.model_dec_seqlen:
                self.piece_admissible_stbars.append([0])
            else:
                _admissible_stbars = []
                for bar in range(len(self.piece_melody_pos[-1])):
                    if len(piece_evs) - self.piece_melody_pos[-1][bar][0] >= 0.5 * self.model_dec_seqlen:
                        _admissible_stbars.append(bar)
                    else:
                        break
                    # if 0.75 * self.model_dec_seqlen <= len(piece_evs) - self.piece_melody_pos[-1][bar][0] < self.model_dec_seqlen - 2:
                    #     _admissible_stbars.append(bar)
                self.piece_admissible_stbars.append(_admissible_stbars)

    def get_sample_from_file(self, piece_idx):
        piece_evs = pickle_load(self.pieces[piece_idx])[2]
        piece_melody_pos = self.piece_melody_pos[piece_idx]
        piece_chord_pos = self.piece_chord_pos[piece_idx]
        assert len(piece_chord_pos) == len(piece_melody_pos)
        # n_bars = len(piece_midi_pos)
        st_bar = random.choice(self.piece_admissible_stbars[piece_idx])
        return piece_evs, piece_melody_pos, piece_chord_pos, st_bar

    def pad_sequence(self, seq, maxlen, pad_value=None):
        if pad_value is None:
            pad_value = self.pad_token

        if len(seq) < maxlen:
            seq.extend([pad_value for _ in range(maxlen - len(seq))])

        return seq

    def make_target_and_mask(self, inp_tokens, melody_pos, chord_pos, st_bar):
        tgt = np.full_like(inp_tokens, fill_value=self.pad_token)
        track_mask = np.zeros_like(inp_tokens)

        for bidx in range(st_bar, len(melody_pos)):
            offset = - melody_pos[st_bar][0] + melody_pos[0][0]

            track_mask[chord_pos[bidx][0] + offset: chord_pos[bidx][1] + offset] = 1
            if bidx != len(melody_pos) - 1:
                tgt[chord_pos[bidx][0] + offset: chord_pos[bidx][1] + offset] = inp_tokens[chord_pos[bidx][0] + 1 + offset:
                                                                                           chord_pos[bidx][1] + 1 + offset]
            else:
                tgt[chord_pos[bidx][0] + offset: chord_pos[bidx][1] - 1 + offset] = inp_tokens[
                                                                                    chord_pos[bidx][0] + 1 + offset:
                                                                                    chord_pos[bidx][1] + offset]
                tgt[chord_pos[bidx][1] - 1 + offset] = self.eos_token

        return tgt, track_mask

    def make_target_and_mask_predict(self, inp_tokens, melody_pos, chord_pos, st_bar):
        # inp:        emotion, key, melody,  chord, melody,  chord
        # tgt:                 key, padding, chord, padding, chord, EOS
        # track_mask: 2,       3,   0,       1,     0,       1
        tgt = np.full_like(inp_tokens, fill_value=self.pad_token)
        track_mask = np.zeros_like(inp_tokens)

        track_mask[0] = 2
        track_mask[1] = 3
        tgt[0] = inp_tokens[1]

        for bidx in range(st_bar, len(melody_pos)):
            offset = - melody_pos[st_bar][0] + melody_pos[0][0]

            track_mask[chord_pos[bidx][0] + offset: chord_pos[bidx][1] + offset] = 1
            if bidx != len(melody_pos) - 1:
                tgt[chord_pos[bidx][0] + offset: chord_pos[bidx][1] + offset] = inp_tokens[chord_pos[bidx][0] + 1 + offset:
                                                                                           chord_pos[bidx][1] + 1 + offset]
            else:
                tgt[chord_pos[bidx][0] + offset: chord_pos[bidx][1] - 1 + offset] = inp_tokens[
                                                                                    chord_pos[bidx][0] + 1 + offset:
                                                                                    chord_pos[bidx][1] + offset]
                tgt[chord_pos[bidx][1] - 1 + offset] = self.eos_token

        return tgt, track_mask

    def __len__(self):
        return len(self.pieces)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        piece_events, melody_pos, chord_pos, st_bar = self.get_sample_from_file(idx)

        # event to index
        st_events = piece_events[:melody_pos[0][0]]
        piece_tokens = convert_event(
            st_events + piece_events[self.piece_melody_pos[idx][st_bar][0]:],
            self.event2idx, to_ndarr=False
        )
        length = len(piece_tokens)

        # padding
        inp = piece_tokens
        if self.pad_to_same:
            inp = self.pad_sequence(piece_tokens, self.model_dec_seqlen)
        inp = np.array(inp, dtype=int)

        # get target and mask
        if self.predict_key:
            target, track_mask = self.make_target_and_mask_predict(inp, melody_pos, chord_pos, st_bar)
        else:
            target, track_mask = self.make_target_and_mask(inp, melody_pos, chord_pos, st_bar)

        # record chord and melody index for accuracy computation
        self.idx2event[self.pad_token] = 'Pad_None'
        target_events = [self.idx2event[i] for i in target]
        target_types = [i.split('_')[0] for i in target_events]
        chord_idx = np.zeros_like(target)
        chord_idx[np.where(np.array(target_types) == 'Chord')[0]] = 1
        melody_idx = np.zeros_like(target)
        melody_idx[np.where(np.array(target_types) == 'Note')[0]] = 1

        inp = inp[:self.model_dec_seqlen]
        target = target[:self.model_dec_seqlen]
        track_mask = track_mask[:self.model_dec_seqlen]
        chord_idx = chord_idx[:self.model_dec_seqlen]
        melody_idx = melody_idx[:self.model_dec_seqlen]

        assert len(inp) == len(target)
        assert len(inp) == len(track_mask)
        assert len(inp) == len(chord_idx)
        assert len(inp) == len(melody_idx)

        return {
            'id': idx,
            'piece_id': self.pieces[idx].split('/')[-1].replace('.pkl', ''),
            'dec_input': inp,
            'dec_target': target,
            'chords_mhot': 0,
            'track_mask': track_mask,
            'length': min(length, self.model_dec_seqlen),
            'chord_idx': chord_idx,
            'melody_idx': melody_idx
        }
