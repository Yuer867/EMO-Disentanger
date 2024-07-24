import time

import scipy
import torch
import numpy as np

from utils import tensor_to_numpy
from convert_key import MINOR_KEY, MAJOR_KEY


########################################
# sampling utilities
########################################
def temperature(logits, temperature):
    try:
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        assert np.count_nonzero(np.isnan(probs)) == 0
    except:
        print('overflow detected, use 128-bit')
        logits = logits.astype(np.float128)
        probs = scipy.special.softmax(logits / temperature)
        probs = probs.astype(float)
        assert np.count_nonzero(np.isnan(probs)) == 0
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


def get_position_idx(event):
    return int(event.split('_')[-1])


########################################
# main inference driver
########################################
def generate_plain_xl(model, event2idx, idx2event, max_bars=160,
                      max_events=2048, primer=None, temp=1.2, top_p=0.9,
                      prompt_bars=None, representation='functional', key_determine=None):
    if primer is None:
        generated = [event2idx['Bar_None']]
        target_bars, generated_bars = max_bars, 0
    else:
        generated = [event2idx[e] for e in primer]
        target_bars, generated_bars = max_bars, prompt_bars if prompt_bars is not None else 0

    device = next(model.parameters()).device
    steps = 0
    time_st = time.time()
    cur_pos = 0
    failed_cnt = 0
    mems = tuple()
    while generated_bars < target_bars:
        if steps == 0:
            dec_input = torch.LongTensor([generated]).to(device)
            dec_input = dec_input.permute(1, 0) if len(generated) > 1 else dec_input
            # print ('[shape]', dec_input.size(), dec_seg_emb.size())
        else:
            dec_input = torch.LongTensor([[generated[-1]]]).to(device)
        # print (dec_input.size(), dec_seg_emb.size())

        # sampling
        logits, mems = model.generate(dec_input, mems)
        logits = tensor_to_numpy(logits)

        if representation in ['functional', 'key'] and len(generated) == 1:
            probs = temperature(logits, temperature=1.1)
            word = nucleus(probs, p=0.97)
            if key_determine == 'rule':
                emotion_label = idx2event[generated[0]].split('_')[1]
                key_event = idx2event[word]
                if key_event.split('_')[0] != 'Key':
                    raise ValueError('[info] key generation failed')
                key_label = key_event.split('_')[1]
                if not match_emotion_key(emotion_label, key_label):
                    continue
            word_event = idx2event[word]
        else:
            probs = temperature(logits, temperature=temp)
            word = nucleus(probs, p=top_p)
            word_event = idx2event[word]
            # print(temp, top_p)

        if 'Key' in word_event:
            print('[info] generated {}, #events = {}'.format(word_event, len(generated)))

        if 'Beat' in word_event:
            event_pos = get_position_idx(word_event)
            if not event_pos >= cur_pos:
                failed_cnt += 1
                print('[info] position not increasing, failed cnt:', failed_cnt)
                if failed_cnt >= 256:
                    print('[FATAL] model stuck, exiting ...')
                    return None, time.time() - time_st
                continue
            else:
                cur_pos = event_pos
                failed_cnt = 0

        if 'Bar' in word_event:
            generated_bars += 1
            cur_pos = 0
            print('[info] generated {} bars, #events = {}'.format(generated_bars, len(generated)))
        if word_event == 'PAD_None':
            continue

        generated.append(word)
        # print ([idx2event[e] for e in generated])
        steps += 1

        if len(generated) > max_events:
            print('[info] max events reached')
            break
        if word_event == 'EOS_None':
            print('[info] gotten eos')
            break

    print('-- generated events:', len(generated))
    print('-- time elapsed: {:.2f} secs'.format(time.time() - time_st))

    return generated[:-1], time.time() - time_st


def match_emotion_key(emotion, key):
    if emotion in ['Q1', 'Q4', 'Positive'] and key in MAJOR_KEY:
        return True
    if emotion in ['Q2', 'Q3', 'Negative'] and key in MINOR_KEY:
        return True
    return False
