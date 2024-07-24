import os
import random
from glob import glob
from copy import deepcopy

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from utils import pickle_load
from convert_key import MINOR_KEY, MAJOR_KEY

IDX_TO_KEY = {
    0: 'A',
    1: 'A#',
    2: 'B',
    3: 'C',
    4: 'C#',
    5: 'D',
    6: 'D#',
    7: 'E',
    8: 'F',
    9: 'F#',
    10: 'G',
    11: 'G#'
}
KEY_TO_IDX = {
    v: k for k, v in IDX_TO_KEY.items()
}

ENC_SEQLEN_ROUNDUP_TABLE = np.array([16 * x for x in range(1, 4)])


def roundup_enc_bar_arr_size(max_bar_len):
    return (
        ENC_SEQLEN_ROUNDUP_TABLE[
            np.searchsorted(ENC_SEQLEN_ROUNDUP_TABLE, max_bar_len)
        ]
    )


def get_chord_tone(chord_event):
    tone = chord_event.split('_')[1]
    return tone


def transpose_chord(chord_event, n_keys):
    if 'N_N' in chord_event:
        return chord_event

    orig_tone = get_chord_tone(chord_event)
    orig_tone_idx = KEY_TO_IDX[orig_tone]
    new_tone_idx = (orig_tone_idx + 12 + n_keys) % 12
    new_chord_event = chord_event.replace(
        '{}_'.format(orig_tone), '{}_'.format(IDX_TO_KEY[new_tone_idx])
    )
    # print ('keys={}. {} --> {}'.format(n_keys, chord_event, new_chord_event))

    return new_chord_event


def check_extreme_pitch(raw_events):
    low, high = 128, 0
    for ev in raw_events:
        if 'Note_Pitch' in ev:
            ev_val = int(ev.split('_')[-1])
            low = min(low, ev_val)
            high = max(high, ev_val)

    return low, high


def transpose_events(raw_events, n_keys):
    transposed_raw_events = []

    for ev in raw_events:
        if 'Note_Pitch' in ev:
            ev_val = int(ev.split('_')[-1])
            transposed_raw_events.append(
                'Note_Pitch_{}'.format(ev_val + n_keys)
            )
        elif 'Chord' in ev:
            # print ('[got chord]')
            transposed_raw_events.append(
                transpose_chord(ev, n_keys)
            )
        else:
            transposed_raw_events.append(ev)
            # print ('keys={}. {} --> {}'.format(n_keys, ev, transposed_raw_events[-1]))

    assert len(transposed_raw_events) == len(raw_events)
    return transposed_raw_events


def convert_event(event_seq, event2idx, to_ndarr=True):
    if isinstance(event_seq[0], dict):
        event_seq = [event2idx['{}_{}'.format(e['name'], e['value'])] for e in event_seq]
    else:
        event_seq = [event2idx[e] for e in event_seq]

    if to_ndarr:
        return np.array(event_seq)
    else:
        return event_seq


def compute_chroma(bar_events, idx2event):
    bar_chroma = np.zeros((12,))
    bar_events = [idx2event[x] for x in deepcopy(bar_events)]

    for e in bar_events:
        if 'Note_Pitch' in e:
            pitch = int(e.split('_')[-1])
            bar_chroma[pitch % 12] += 1

    bar_chroma = bar_chroma / (np.linalg.norm(bar_chroma) + 1e-8)
    # print (bar_events)
    # print (bar_chroma)
    # print ('-------------------------------------\n')
    return bar_chroma


def compute_groove(bar_events, idx2event):
    bar_groove = np.zeros((16,))
    bar_events = [idx2event[x] for x in deepcopy(bar_events)]

    for e in bar_events:
        if 'Beat' in e:
            position = int(e.split('_')[-1])
            bar_groove[position] = 1.

    # bar_chroma = bar_chroma / ( np.linalg.norm(bar_chroma) + 1e-8 )
    # print (bar_events)
    # print (bar_chroma)
    # print ('-------------------------------------\n')
    return bar_groove


def compute_feature_masks(bar_events, idx2event, max_bar_len):
    chroma_mask = np.zeros((max_bar_len,), dtype=float)
    groove_mask = np.zeros((max_bar_len,), dtype=float)
    bar_events = [idx2event[x] for x in deepcopy(bar_events)]

    for i, e in enumerate(bar_events):
        if i >= max_bar_len:
            break

        if 'Note_Pitch' in e:
            chroma_mask[i] = 1.
        if 'Note_Duration' in e:
            groove_mask[i] = 1.
        if 'Beat' in e:
            groove_mask[i] = 1.

    return chroma_mask, groove_mask


class SkylineFullSongTransformerDataset(Dataset):
    def __init__(self, data_dir, vocab_file,
                 model_enc_seqlen=32, model_dec_seqlen=2400, model_max_bars=192,
                 pieces=[], do_augment=True, augment_range=range(-6, 7),
                 min_pitch=48, max_pitch=108, max_n_seg=1,
                 convert_dict_event=False, extend_enc_seq=False):
        self.vocab_file = vocab_file
        self.read_vocab()

        self.data_dir = data_dir
        self.pieces = pieces
        self.model_enc_seqlen = model_enc_seqlen
        self.model_dec_seqlen = model_dec_seqlen
        self.model_max_bars = model_max_bars

        self.max_n_seg = max_n_seg
        self.build_dataset()
        self.register_segments()

        self.do_augment = do_augment
        self.augment_range = augment_range
        self.min_pitch, self.max_pitch = min_pitch, max_pitch

        # if self.do_augment:
        #     self._default_collate_keys = [
        #         'id', 'piece_id', 'st_seg', 'transpose_keys', 'n_seg'
        #     ]
        # else:
        self._default_collate_keys = [
            'id', 'piece_id', 'st_seg', 'n_seg'
        ]

        self.convert_dict_event = convert_dict_event
        self.extend_enc_seq = extend_enc_seq

    def collate_fn(self, batch):
        batch_n_segs = max([samp['n_seg'] for samp in batch])
        max_enc_len = [0 for seg in range(self.max_n_seg)]
        max_dec_bars = [0 for seg in range(self.max_n_seg)]

        for seg in range(batch_n_segs):
            for samp_idx in range(len(batch)):
                if 'enc_seg_len_{}'.format(seg) in batch[samp_idx]:
                    max_enc_len[seg] = \
                        max(max_enc_len[seg], batch[samp_idx]['enc_seg_len_{}'.format(seg)])
                    max_dec_bars[seg] = \
                        max(max_dec_bars[seg], len(batch[samp_idx]['dec_bar_pos_{}'.format(seg)]))

        default_collate_keys = deepcopy(self._default_collate_keys)
        for seg in range(batch_n_segs):
            for samp_idx in range(len(batch)):
                if 'dec_inp_{}'.format(seg) in batch[samp_idx]:
                    if batch[samp_idx]['dec_seg_len_{}'.format(seg)] < self.model_dec_seqlen:
                        pad_len = self.model_dec_seqlen - batch[samp_idx]['dec_seg_len_{}'.format(seg)]
                        batch[samp_idx]['dec_inp_{}'.format(seg)] = np.concatenate((
                            batch[samp_idx]['dec_inp_{}'.format(seg)],
                            np.full((pad_len,), fill_value=self.pad_token)
                        ))
                        batch[samp_idx]['dec_tgt_{}'.format(seg)] = np.concatenate((
                            batch[samp_idx]['dec_tgt_{}'.format(seg)],
                            np.full((pad_len,), fill_value=self.pad_token)
                        ))
                        batch[samp_idx]['inp_chord_{}'.format(seg)] = np.concatenate((
                            batch[samp_idx]['inp_chord_{}'.format(seg)],
                            np.full((pad_len,), fill_value=self.pad_token)
                        ))
                        batch[samp_idx]['inp_melody_{}'.format(seg)] = np.concatenate((
                            batch[samp_idx]['inp_melody_{}'.format(seg)],
                            np.full((pad_len,), fill_value=self.pad_token)
                        ))
                    if len(batch[samp_idx]['dec_bar_pos_{}'.format(seg)]) < max_dec_bars[seg]:
                        pad_len = max_dec_bars[seg] - len(batch[samp_idx]['dec_bar_pos_{}'.format(seg)])
                        batch[samp_idx]['dec_bar_pos_{}'.format(seg)] = np.concatenate((
                            batch[samp_idx]['dec_bar_pos_{}'.format(seg)],
                            np.full((pad_len,), fill_value=-1)
                        ))
                else:
                    batch[samp_idx]['dec_inp_{}'.format(seg)] = np.full((self.model_dec_seqlen,),
                                                                        fill_value=self.pad_token)
                    batch[samp_idx]['dec_tgt_{}'.format(seg)] = np.full((self.model_dec_seqlen,),
                                                                        fill_value=self.pad_token)
                    batch[samp_idx]['inp_chord_{}'.format(seg)] = np.full((self.model_dec_seqlen,),
                                                                          fill_value=self.pad_token)
                    batch[samp_idx]['inp_melody_{}'.format(seg)] = np.full((self.model_dec_seqlen,),
                                                                           fill_value=self.pad_token)
                    batch[samp_idx]['dec_bar_pos_{}'.format(seg)] = np.full((max_dec_bars[seg],), fill_value=-1)
                    batch[samp_idx]['dec_seg_len_{}'.format(seg)] = 0

            default_collate_keys.extend([
                'dec_inp_{}'.format(seg),
                'dec_tgt_{}'.format(seg),
                'dec_bar_pos_{}'.format(seg),
                'dec_seg_len_{}'.format(seg),
                'inp_chord_{}'.format(seg),
                'inp_melody_{}'.format(seg)
            ])

        # print (default_collate_keys)
        _batch = dict()
        for key in default_collate_keys:
            _batch[key] = default_collate([samp[key] for samp in batch])

        _ext_batch = dict()
        for seg in range(batch_n_segs):
            enc_bar_st_idx = [0]
            _concat_enc_inp = []
            _concat_enc_pad = []
            _concat_enc_chroma = []
            _concat_enc_groove = []
            _concat_enc_chroma_mask = []
            _concat_enc_groove_mask = []

            for samp_idx in range(len(batch)):
                _e_slen_key = 'enc_seg_len_{}'.format(seg)
                _e_pad_key = 'enc_padding_mask_{}'.format(seg)
                _e_inp_key = 'enc_inp_{}'.format(seg)
                _e_chroma_key = 'enc_chroma_{}'.format(seg)
                _e_groove_key = 'enc_groove_{}'.format(seg)
                _e_chroma_mask_key = 'enc_chroma_mask_{}'.format(seg)
                _e_groove_mask_key = 'enc_groove_mask_{}'.format(seg)

                if _e_slen_key in batch[samp_idx]:
                    samp_n_bars = len(batch[samp_idx][_e_inp_key])
                    enc_bar_st_idx.append(enc_bar_st_idx[-1] + samp_n_bars)

                    if batch[samp_idx][_e_slen_key] < max_enc_len[seg]:
                        pad_len = max_enc_len[seg] - batch[samp_idx][_e_slen_key]
                        batch[samp_idx][_e_inp_key] = np.concatenate((
                            batch[samp_idx][_e_inp_key],
                            np.full((samp_n_bars, pad_len), fill_value=self.pad_token)
                        ), axis=-1)
                        batch[samp_idx][_e_pad_key] = np.concatenate((
                            batch[samp_idx][_e_pad_key],
                            np.full((samp_n_bars, pad_len), fill_value=True)
                        ), axis=-1)
                        batch[samp_idx][_e_chroma_mask_key] = np.concatenate((
                            batch[samp_idx][_e_chroma_mask_key],
                            np.zeros((samp_n_bars, pad_len), dtype=float)
                        ), axis=-1)
                        batch[samp_idx][_e_groove_mask_key] = np.concatenate((
                            batch[samp_idx][_e_groove_mask_key],
                            np.zeros((samp_n_bars, pad_len), dtype=float)
                        ), axis=-1)

                    _concat_enc_inp.append(batch[samp_idx][_e_inp_key])
                    _concat_enc_pad.append(batch[samp_idx][_e_pad_key])
                    _concat_enc_chroma.append(batch[samp_idx][_e_chroma_key])
                    _concat_enc_groove.append(batch[samp_idx][_e_groove_key])
                    _concat_enc_chroma_mask.append(batch[samp_idx][_e_chroma_mask_key])
                    _concat_enc_groove_mask.append(batch[samp_idx][_e_groove_mask_key])
                else:
                    enc_bar_st_idx.append(enc_bar_st_idx[-1])

            if len(_concat_enc_inp) > 1:
                batch_seg_enc_inp = np.concatenate(tuple(_concat_enc_inp))
                batch_seg_enc_pad = np.concatenate(tuple(_concat_enc_pad))
                batch_seg_enc_chroma = np.concatenate(tuple(_concat_enc_chroma))
                batch_seg_enc_groove = np.concatenate(tuple(_concat_enc_groove))
                batch_seg_enc_chroma_mask = np.concatenate(tuple(_concat_enc_chroma_mask))
                batch_seg_enc_groove_mask = np.concatenate(tuple(_concat_enc_groove_mask))
            else:
                batch_seg_enc_inp = _concat_enc_inp[0]
                batch_seg_enc_pad = _concat_enc_pad[0]
                batch_seg_enc_chroma = _concat_enc_chroma[0]
                batch_seg_enc_groove = _concat_enc_groove[0]
                batch_seg_enc_chroma_mask = _concat_enc_chroma_mask[0]
                batch_seg_enc_groove_mask = _concat_enc_groove_mask[0]

            _ext_batch['enc_inp_{}'.format(seg)] = batch_seg_enc_inp
            _ext_batch['enc_padding_mask_{}'.format(seg)] = batch_seg_enc_pad
            _ext_batch['enc_chroma_{}'.format(seg)] = batch_seg_enc_chroma
            _ext_batch['enc_groove_{}'.format(seg)] = batch_seg_enc_groove
            _ext_batch['enc_chroma_mask_{}'.format(seg)] = batch_seg_enc_chroma_mask
            _ext_batch['enc_groove_mask_{}'.format(seg)] = batch_seg_enc_groove_mask
            _ext_batch['enc_bar_st_idx_{}'.format(seg)] = np.array(enc_bar_st_idx)
            # print (_ext_batch['enc_bar_st_idx_{}'.format(seg)], batch_seg_enc_inp.shape)

        for k, v in _ext_batch.items():
            _ext_batch[k] = torch.as_tensor(v)

        _batch.update(_ext_batch)

        return _batch

    def read_vocab(self):
        vocab = pickle_load(self.vocab_file)[0]
        self.idx2event = pickle_load(self.vocab_file)[1]
        orig_vocab_size = len(vocab)
        self.event2idx = vocab
        self.bar_token = self.event2idx['Bar_None']
        self.eos_token = self.event2idx['EOS_None']
        self.pad_token = orig_vocab_size
        self.event2idx['PAD_None'] = self.pad_token
        self.vocab_size = self.pad_token + 1

    def build_dataset(self):
        if not self.pieces:
            self.pieces = sorted(glob(os.path.join(self.data_dir, '*.pkl')))
        else:
            self.pieces = sorted([
                os.path.join(self.data_dir, p) for p in self.pieces \
                if os.path.exists(os.path.join(self.data_dir, p))
            ])

        self.piece_bar_pos = []

        for i, p in enumerate(self.pieces):
            bar_pos, p_evs = pickle_load(p)
            if not i % 200:
                print('[preparing data] now at #{}'.format(i))
            if bar_pos[-1] == len(p_evs):
                print('piece {}, got appended bar markers'.format(p))
                bar_pos = bar_pos[:-1]

            # remove empty bar
            if len(p_evs[bar_pos[-1]:]) == 2:
                p_evs = p_evs[: bar_pos[-1]]
                bar_pos = bar_pos[:-1]

            if len(bar_pos) <= self.model_max_bars:
                # remove <EOS>
                bar_pos.append(len(p_evs) - 1)
            else:
                bar_pos = bar_pos[:self.model_max_bars + 1]

            self.piece_bar_pos.append(bar_pos)

    def register_segments(self):
        self.total_segs = 0
        self.piece_segments = []

        for p, p_bar_pos in enumerate(self.piece_bar_pos):
            p_segment = []
            st_bar = 0

            for b, b_start in enumerate(p_bar_pos):
                if b == len(p_bar_pos) - 1:
                    break
                if p_bar_pos[b + 1] - p_bar_pos[st_bar] > self.model_dec_seqlen - 1:
                    if b > st_bar:
                        p_segment.append((st_bar, b))
                        st_bar = b
                        break

            if len(p_segment) < self.max_n_seg:
                p_segment.append((st_bar, len(p_bar_pos) - 1))
            self.total_segs += len(p_segment)
            self.piece_segments.append(p_segment)

    def get_sample_from_file(self, piece_idx):
        piece_evs = pickle_load(self.pieces[piece_idx])[1]
        if isinstance(piece_evs, np.ndarray):
            piece_evs = piece_evs.tolist()

        piece_evs = piece_evs[: self.piece_bar_pos[piece_idx][-1]]

        if self.convert_dict_event:
            piece_evs = ['{}_{}'.format(x['name'], x['value']) for x in piece_evs]

        piece_evs_orig = deepcopy(piece_evs)

        if self.max_n_seg is None:
            st_seg = 0
            ed_seg = len(self.piece_segments[piece_idx]) - 1
            if len(self.piece_bar_pos[piece_idx]) <= self.model_max_bars:
                piece_evs.append('EOS_None')
            else:
                piece_evs.append('Bar_None')

        else:
            piece_seg = self.piece_segments[piece_idx]
            # if len(self.piece_segments[ piece_idx ]) <= self.max_n_seg:
            # if len(piece_seg) <= self.max_n_seg:
            st_seg = 0
            ed_seg = 0
            if len(self.piece_bar_pos[piece_idx]) <= self.model_max_bars:
                piece_evs.append('EOS_None')
            else:
                # print ('[ -- long piece -- ]', len(self.piece_bar_pos[piece_idx]))
                piece_evs.append('Bar_None')
            # print ('[id {:04d}] short !!'.format(piece_idx))
            ext_enc_bar_evs = None

        if not self.extend_enc_seq:
            return piece_evs, st_seg, ed_seg
        else:
            return piece_evs, ext_enc_bar_evs, st_seg, ed_seg

    def pitch_augment(self, piece_events):
        bar_min_pitch, bar_max_pitch = check_extreme_pitch(piece_events)
        # print (bar_min_pitch, bar_max_pitch)

        n_keys = random.choice(self.augment_range)
        while bar_min_pitch + n_keys < self.min_pitch or bar_max_pitch + n_keys > self.max_pitch:
            n_keys = random.choice(self.augment_range)

        augmented_piece_events = transpose_events(piece_events, n_keys)
        return augmented_piece_events, n_keys

    def key_augment(self, piece_events):
        if piece_events[1].split('_')[0] != 'Key':
            raise ValueError('wrong key event')
        if piece_events[1].split('_')[1] in MAJOR_KEY:
            new_key = random.choice(MAJOR_KEY)
            piece_events[1] = 'Key_{}'.format(new_key)
        if piece_events[1].split('_')[1] in MINOR_KEY:
            new_key = random.choice(MINOR_KEY)
            piece_events[1] = 'Key_{}'.format(new_key)
        return piece_events

    def get_decoder_input_data(self, piece_idx, bar_positions, piece_events, piece_tokens, st_seg, ed_seg):
        inp_segments = []
        tgt_segments = []
        segments_bar_pos = []
        segments_len = []

        piece_types = [i.split('_')[0] for i in piece_events]
        inp_chord = []
        inp_melody = []

        sample_segs = self.piece_segments[piece_idx][st_seg: ed_seg + 1]
        sample_st_idx = bar_positions[sample_segs[0][0]]
        # sample_st_idx = 0
        # print ('[info] # segs:', len(sample_segs))

        for seg, (seg_st_bar, seg_ed_bar) in enumerate(sample_segs):
            seg_st_idx, seg_ed_idx = \
                bar_positions[seg_st_bar] - sample_st_idx, bar_positions[seg_ed_bar] - sample_st_idx + 1

            # print (piece_events[seg_st_idx : seg_ed_idx + 1][ -8 : ])

            inp_segments.append(
                np.array(piece_tokens[seg_st_idx: seg_ed_idx])
            )
            tgt_segments.append(
                np.array(piece_tokens[seg_st_idx + 1: seg_ed_idx + 1])
            )
            segments_bar_pos.append(
                np.array(bar_positions[seg_st_bar: seg_ed_bar + 1])
            )
            segments_len.append(len(inp_segments[-1]))

            chord_idx = np.zeros_like(piece_tokens[seg_st_idx + 1: seg_ed_idx + 1])
            chord_idx[np.where(np.array(piece_types[seg_st_idx + 1: seg_ed_idx + 1]) == 'Chord')[0]] = 1
            inp_chord.append(chord_idx)

            melody_idx = np.zeros_like(piece_tokens[seg_st_idx + 1: seg_ed_idx + 1])
            melody_idx[np.where(np.array(piece_types[seg_st_idx + 1: seg_ed_idx + 1]) == 'Note')[0]] = 1
            inp_melody.append(melody_idx)

            if len(inp_segments[-1]) > self.model_dec_seqlen:
                inp_segments[-1] = inp_segments[-1][:self.model_dec_seqlen]
                tgt_segments[-1] = tgt_segments[-1][:self.model_dec_seqlen]
                segments_bar_pos[-1][-1] = self.model_dec_seqlen
                inp_chord[-1] = inp_chord[-1][:self.model_dec_seqlen]
                inp_melody[-1] = inp_melody[-1][:self.model_dec_seqlen]

            assert len(inp_segments[-1]) == len(tgt_segments[-1])
            # print (segments_bar_pos)
            # assert segments_bar_pos[-1][-1] == len(inp_segments[-1]), '{} != {}'.format(segments_bar_pos[-1][-1], len(inp_segments[-1]))

        return inp_segments, tgt_segments, segments_bar_pos, segments_len, inp_chord, inp_melody

    def pad_or_truncate_sequence(self, seq, maxlen, pad_value=None):
        if pad_value is None:
            pad_value = self.pad_token

        if len(seq) < maxlen:
            seq.extend([pad_value for _ in range(maxlen - len(seq))])
        else:
            seq = seq[:maxlen]

        return seq

    def get_encoder_input_data(self, piece_idx, bar_positions, piece_events, st_seg, ed_seg, ext_enc_bar_evs=None):
        inp_segments = []
        segments_padding_mask = []
        segments_enc_len = []
        segments_chroma_vecs = []
        segments_groove_vecs = []
        segments_chroma_masks = []
        segments_groove_masks = []

        bar_positions = np.array(bar_positions)
        sample_segs = self.piece_segments[piece_idx][st_seg: ed_seg + 1]
        # sample_st_idx = bar_positions[ sample_segs[0][0] ]
        sample_st_idx = 0

        for seg, (seg_st_bar, seg_ed_bar) in enumerate(sample_segs):

            if self.extend_enc_seq and seg < len(sample_segs) - 1:
                seg_ed_bar += 1

            _segment_max_bar_len = min(
                self.model_enc_seqlen,
                max(bar_positions[seg_st_bar + 1: seg_ed_bar + 1] - bar_positions[seg_st_bar: seg_ed_bar])
            )
            if self.extend_enc_seq and seg == len(sample_segs) - 1 and ext_enc_bar_evs is not None:
                _segment_max_bar_len = min(
                    self.model_enc_seqlen,
                    max(_segment_max_bar_len, len(ext_enc_bar_evs))
                )

            segment_max_bar_len = roundup_enc_bar_arr_size(_segment_max_bar_len)
            n_segment_bars = seg_ed_bar - seg_st_bar
            if self.extend_enc_seq and seg == len(sample_segs) - 1 and ext_enc_bar_evs is not None:
                n_segment_bars += 1
            # print (_segment_max_bar_len, segment_max_bar_len)

            seg_chroma_mask = np.zeros((n_segment_bars, segment_max_bar_len), dtype=float)
            seg_groove_mask = np.zeros((n_segment_bars, segment_max_bar_len), dtype=float)

            seg_padding_mask = np.ones((n_segment_bars, segment_max_bar_len), dtype=bool)
            seg_padded_input = np.full((n_segment_bars, segment_max_bar_len), dtype=int, fill_value=self.pad_token)
            seg_chroma_vecs = np.zeros((n_segment_bars, 12))
            seg_groove_vecs = np.zeros((n_segment_bars, 16))

            seg_bar_positions = bar_positions[seg_st_bar: seg_ed_bar + 1] - sample_st_idx
            for b, (st, ed) in enumerate(zip(seg_bar_positions[:-1], seg_bar_positions[1:])):
                seg_padding_mask[b, : (ed - st)] = False
                seg_chroma_vecs[b, :] = compute_chroma(piece_events[st: ed], self.idx2event)
                seg_groove_vecs[b, :] = compute_groove(piece_events[st: ed], self.idx2event)
                seg_chroma_mask[b, :], seg_groove_mask[b, :] = compute_feature_masks(piece_events[st: ed],
                                                                                     self.idx2event,
                                                                                     segment_max_bar_len)

                within_bar_events = self.pad_or_truncate_sequence(piece_events[st: ed], segment_max_bar_len,
                                                                  self.pad_token)
                assert len(piece_events[st: ed]) == ed - st
                within_bar_events = np.array(within_bar_events)
                seg_padded_input[b, :] = within_bar_events

            if self.extend_enc_seq and seg == len(sample_segs) - 1 and ext_enc_bar_evs is not None:
                assert n_segment_bars == len(seg_bar_positions)
                seg_padding_mask[-1, : len(ext_enc_bar_evs)] = False
                within_bar_events = self.pad_or_truncate_sequence(
                    ext_enc_bar_evs, segment_max_bar_len, self.pad_token
                )
                within_bar_events = np.array(within_bar_events)
                seg_padded_input[-1, :] = within_bar_events

            inp_segments.append(seg_padded_input)
            segments_padding_mask.append(seg_padding_mask)
            segments_enc_len.append(segment_max_bar_len)
            segments_chroma_vecs.append(seg_chroma_vecs)
            segments_groove_vecs.append(seg_groove_vecs)
            segments_chroma_masks.append(seg_chroma_mask)
            segments_groove_masks.append(seg_groove_mask)

        return inp_segments, segments_padding_mask, segments_enc_len, segments_chroma_vecs, segments_groove_vecs, segments_chroma_masks, segments_groove_masks

    def __len__(self):
        return len(self.pieces)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ext_enc_bar_evs = None
        if not self.extend_enc_seq:
            piece_events, st_seg, ed_seg = self.get_sample_from_file(idx)
        else:
            piece_events, ext_enc_bar_evs, st_seg, ed_seg = self.get_sample_from_file(idx)
            piece_events_len = len(piece_events)

        if self.do_augment:
            # if ext_enc_bar_evs is None:
            #     piece_events, transpose_keys = self.pitch_augment(piece_events)
            # else:
            #     ext_events_len = len(ext_enc_bar_evs)
            #     _cat_piece_events = piece_events + ext_enc_bar_evs
            #     _cat_piece_events, transpose_keys = self.pitch_augment(_cat_piece_events)
            #     piece_events, ext_enc_bar_evs = \
            #         _cat_piece_events[: piece_events_len], _cat_piece_events[piece_events_len:]
            #     assert len(piece_events) == piece_events_len and len(ext_enc_bar_evs) == ext_events_len
            piece_events = self.key_augment(piece_events)

        piece_tokens = convert_event(piece_events, self.event2idx, to_ndarr=False)
        if ext_enc_bar_evs is not None:
            ext_enc_bar_evs = convert_event(ext_enc_bar_evs, self.event2idx, to_ndarr=False)
        bar_pos = self.piece_bar_pos[idx]

        data_dict = dict()
        dec_inp, dec_tgt, dec_bar_positions, dec_seg_len, inp_chord, inp_melody = self.get_decoder_input_data(
            idx, bar_pos, piece_events, piece_tokens, st_seg, ed_seg
        )
        data_dict['id'] = idx
        data_dict['piece_id'] = self.pieces[idx].split('/')[-1].replace('.pkl', '')
        data_dict['st_seg'] = st_seg
        # if self.do_augment:
        #     data_dict['transpose_keys'] = transpose_keys
        data_dict['n_seg'] = len(dec_inp)

        for s in range(data_dict['n_seg']):
            data_dict['dec_inp_{}'.format(s)] = dec_inp[s]
            data_dict['dec_tgt_{}'.format(s)] = dec_tgt[s]
            data_dict['dec_bar_pos_{}'.format(s)] = dec_bar_positions[s]
            data_dict['dec_seg_len_{}'.format(s)] = dec_seg_len[s]
            data_dict['inp_chord_{}'.format(s)] = inp_chord[s]
            data_dict['inp_melody_{}'.format(s)] = inp_melody[s]

        if not self.extend_enc_seq:
            enc_inp, enc_padding_mask, enc_seg_len, enc_chroma_vec, enc_groove_vec, \
            enc_chroma_mask, enc_groove_mask = self.get_encoder_input_data(
                idx, bar_pos, piece_tokens, st_seg, ed_seg
            )
        else:
            enc_inp, enc_padding_mask, enc_seg_len, enc_chroma_vec = self.get_encoder_input_data(
                idx, bar_pos, piece_tokens, st_seg, ed_seg,
                ext_enc_bar_evs=ext_enc_bar_evs
            )
        assert len(enc_inp) == len(enc_padding_mask) and len(enc_inp) == data_dict['n_seg']
        for s in range(data_dict['n_seg']):
            data_dict['enc_inp_{}'.format(s)] = enc_inp[s]
            data_dict['enc_padding_mask_{}'.format(s)] = enc_padding_mask[s]
            data_dict['enc_seg_len_{}'.format(s)] = enc_seg_len[s]
            data_dict['enc_chroma_{}'.format(s)] = enc_chroma_vec[s]
            data_dict['enc_groove_{}'.format(s)] = enc_groove_vec[s]
            data_dict['enc_chroma_mask_{}'.format(s)] = enc_chroma_mask[s]
            data_dict['enc_groove_mask_{}'.format(s)] = enc_groove_mask[s]

            # assert len(data_dict['enc_inp_{}'.format(s)]) == len(data_dict['dec_bar_pos_{}'.format(s)]) - 1

        return data_dict
