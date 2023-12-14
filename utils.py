"""
Created on Thu Dec 14 10:12:13 2023
@author: Shulei Ji
"""
import time
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def timeSince(since):
    now=time.time()
    s=now-since
    h=math.floor(s/3600)
    s-=h*3600
    m=math.floor(s/60)
    s-=m*60

    return '%dh_%dm_%ds' % (h, m, s)


def normalize(x):
    x -= x.mean()
    x /= (x.std()+1e-8)
    return x


BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4


# refer to https://github.com/annahung31/EMOPIA/blob/main/workspace/transformer/utils.py
def write_midi_new(words, path_outfile, word2event):
    class_keys = word2event.keys()
    # words = np.load(path_infile)
    midi_obj = miditoolkit.midi.parser.MidiFile()

    bar_cnt = 0
    cur_pos = 0
    last_pos = 0
    all_notes = []

    cnt_error = 0
    for i in range(len(words)):
        vals = []
        for kidx, key in enumerate(class_keys):
            vals.append(word2event[key][words[i][kidx]])
        # print(vals)

        if vals[3] == 'Metrical':
            if vals[2]==0:
                pass
            elif vals[2] == 'Bar':
                bar_cnt += 1
            elif 'Beat' in vals[2]:
                beat_pos = int(vals[2].split('_')[1])
                cur_pos = bar_cnt * BAR_RESOL + beat_pos * TICK_RESOL
                # chord
                if vals[1] != 'CONTI' and vals[1] != 0:
                    midi_obj.markers.append(
                        Marker(text=str(vals[1]), time=cur_pos))

                if vals[0] != 'CONTI' and vals[0] != 0:
                    tempo = int(vals[0].split('_')[-1])
                    midi_obj.tempo_changes.append(
                        TempoChange(tempo=tempo, time=cur_pos))
            else:
                pass
        elif vals[3] == 'Note':

            try:
                pitch = vals[4].split('_')[-1]
                duration = vals[5].split('_')[-1]
                velocity = vals[6].split('_')[-1]
                if last_pos>0 and (cur_pos-last_pos)>0:
                    offset=(cur_pos-last_pos)//BAR_RESOL
                    if offset!=0:
                        print("-------------offset: ",offset)
                        bar_cnt-=offset
                        cur_pos-=(offset*BAR_RESOL)
                if int(duration) == 0:
                    duration = 60
                end = cur_pos + int(duration)
                last_pos=end
                all_notes.append(
                    Note(
                        pitch=int(pitch),
                        start=cur_pos,
                        end=end,
                        velocity=int(velocity))
                )
            except:
                continue
        else:
            pass

    # save midi
    piano_track = Instrument(0, is_drum=False, name='piano')
    piano_track.notes = all_notes
    midi_obj.instruments = [piano_track]
    midi_obj.dump(path_outfile)


def write_midi(words, path_outfile, word2event):
    class_keys = word2event.keys()
    # words = np.load(path_infile)
    midi_obj = miditoolkit.midi.parser.MidiFile()

    bar_cnt = 0
    cur_pos = 0

    all_notes = []

    cnt_error = 0
    for i in range(len(words)):
        vals = []
        for kidx, key in enumerate(class_keys):
            vals.append(word2event[key][words[i][kidx]])
        # print(vals)

        if vals[3] == 'Metrical':
            if vals[2]==0:
                pass
            elif vals[2] == 'Bar':
                bar_cnt += 1
            elif 'Beat' in vals[2]:
                beat_pos = int(vals[2].split('_')[1])
                cur_pos = bar_cnt * BAR_RESOL + beat_pos * TICK_RESOL

                # chord
                if vals[1] != 'CONTI' and vals[1] != 0:
                    midi_obj.markers.append(
                        Marker(text=str(vals[1]), time=cur_pos))

                if vals[0] != 'CONTI' and vals[0] != 0:
                    tempo = int(vals[0].split('_')[-1])
                    midi_obj.tempo_changes.append(
                        TempoChange(tempo=tempo, time=cur_pos))
            else:
                pass
        elif vals[3] == 'Note':

            try:
                pitch = vals[4].split('_')[-1]
                duration = vals[5].split('_')[-1]
                velocity = vals[6].split('_')[-1]
                if int(duration) == 0:
                    duration = 60
                end = cur_pos + int(duration)

                all_notes.append(
                    Note(
                        pitch=int(pitch),
                        start=cur_pos,
                        end=end,
                        velocity=int(velocity))
                )
            except:
                continue
        else:
            pass

    # save midi
    piano_track = Instrument(0, is_drum=False, name='piano')
    piano_track.notes = all_notes
    midi_obj.instruments = [piano_track]
    midi_obj.dump(path_outfile)


################################################################################
# Sampling
# refer to https://github.com/annahung31/EMOPIA/blob/main/workspace/transformer/utils.py
################################################################################

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    return y


def softmax_with_temperature(logits, temperature):
    c = np.max(logits)
    probs = np.exp((logits-c) / temperature) / np.sum(np.exp((logits-c) / temperature))
    if np.isnan(probs).any():
        return None
    else:
        return probs


def gumbel_softmax(logits, temperature):
    return F.gumbel_softmax(logits, tau=temperature, hard=True)


def weighted_sampling(probs):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    word = np.random.choice(sorted_index, size=1, p=sorted_probs)[0]
    return word


# -- nucleus -- #
def nucleus(probs, p):
    probs /= (sum(probs) + 1e-5)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][0] + 1
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:]
    candi_probs = [probs[i] for i in candi_index]
    candi_probs /= sum(candi_probs)
    try:
        word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    except:
        ipdb.set_trace()
    return word


def sampling(logit, p=None, t=1.0, is_training=False):
    if is_training:
        logit = logit.squeeze()
        probs = gumbel_softmax(logits=logit, temperature=t)

        return torch.argmax(probs)

    else:
        logit = logit.squeeze().cpu().numpy()
        probs = softmax_with_temperature(logits=logit, temperature=t)

        if probs is None:
            return None

        if p is not None:
            cur_word = nucleus(probs, p=p)

        else:
            cur_word = weighted_sampling(probs)
        return cur_word


