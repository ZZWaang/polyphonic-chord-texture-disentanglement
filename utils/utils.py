import math

import pretty_midi as pm
import copy
import pretty_midi as pyd
import os
import random
from typing import Sequence
import numpy as np
import torch


def bpm_to_rate(bpm):
    return 60 / bpm


def ext_nmat_to_nmat(ext_nmat):
    nmat = np.zeros((ext_nmat.shape[0], 4))
    nmat[:, 0] = ext_nmat[:, 0] + ext_nmat[:, 1] / ext_nmat[:, 2]
    nmat[:, 1] = ext_nmat[:, 3] + ext_nmat[:, 4] / ext_nmat[:, 5]
    nmat[:, 2] = ext_nmat[:, 6]
    nmat[:, 3] = ext_nmat[:, 7]
    return nmat


# def nmat_to_pr(nmat, num_step=32):
#     pr = np.zeros((num_step, 128))
#     for s, e, p, v in pr:
#         pr[s, p]

def nmat_to_notes(nmat, start, bpm):
    notes = []
    for s, e, p, v in nmat:
        assert s < e
        assert 0 <= p < 128
        assert 0 <= v < 128
        s = start + s * bpm_to_rate(bpm)
        e = start + e * bpm_to_rate(bpm)
        notes.append(pm.Note(int(v), int(p), s, e))
    return notes


def ext_nmat_to_pr(ext_nmat, num_step=32):
    # [start measure, no, deno, .., .., .., pitch, vel]
    # This is not RIGHT in general. Only works for 2-bar 4/4 music for now.
    pr = np.zeros((num_step, 128))
    if ext_nmat is not None:
        for (sb, sq, sde, eb, eq, ede, p, v) in ext_nmat:
            s_ind = int(sb * sde + sq)
            e_ind = int(eb * ede + eq)
            p = int(p)
            pr[s_ind, p] = 2
            pr[s_ind + 1: e_ind, p] = 1  # note not including the last ind
    return pr


def ext_nmat_to_mel_pr(ext_nmat, num_step=32):
    # [start measure, no, deno, .., .., .., pitch, vel]
    # This is not RIGHT in general. Only works for 2-bar 4/4 music for now.
    pr = np.zeros((num_step, 130))
    pr[:, 129] = 1
    if ext_nmat is not None:
        for (sb, sq, sde, eb, eq, ede, p, v) in ext_nmat:
            s_ind = int(sb * sde + sq)
            e_ind = int(eb * ede + eq)
            p = int(p)
            pr[s_ind, p] = 1
            pr[s_ind: e_ind, 129] = 0
            pr[s_ind + 1: e_ind, 128] = 1  # note not including the last ind
    return pr


def augment_pr(pr, shift=0):
    # it assures to work on single pr
    # for an array of pr, should double-check
    return np.roll(pr, shift, axis=-1)


def augment_mel_pr(pr, shift=0):
    # it only works on single mel_pr. Not on array of it.
    pitch_part = np.roll(pr[:, 0: 128], shift, axis=-1)
    control_part = pr[:, 128:]
    augmented_pr = np.concatenate([pitch_part, control_part], axis=-1)
    return augmented_pr


def pr_to_onehot_pr(pr):
    onset_data = pr[:, :] == 2
    sustain_data = pr[:, :] == 1
    silence_data = pr[:, :] == 0
    pr = np.stack([onset_data, sustain_data, silence_data],
                  axis=-1).astype(np.int64)
    return pr


def piano_roll_to_target(pr):
    #  pr: (32, 128, 3), dtype=bool

    # Assume that "not (first_layer or second layer) = third_layer"
    pr[:, :, 1] = np.logical_not(np.logical_or(pr[:, :, 0], pr[:, :, 2]))
    # To int dtype can make addition work
    pr = pr.astype(int)
    # Initialize a matrix to store the duration of a note on the (32, 128) grid
    pr_matrix = np.zeros((32, 128))

    for i in range(32-1, -1, -1):
        # At each iteration
        # 1. Assure that the second layer accumulates the note duration
        # 2. collect the onset notes in time step i, and mark it on the matrix.

        # collect
        onset_idx = np.where(pr[i, :, 0] == 1)[0]
        pr_matrix[i, onset_idx] = pr[i, onset_idx, 1] + 1
        if i == 0:
            break
        # Accumulate
        # pr[i - 1, :, 1] += pr[i, :, 1]
        # pr[i - 1, onset_idx, 1] = 0  # the onset note should be set 0.

        pr[i, onset_idx, 1] = 0  # the onset note should be set 0.
        pr[i - 1, :, 1] += pr[i, :, 1]
    return pr_matrix


def target_to_3dtarget(pr_mat, max_note_count=11, max_pitch=107, min_pitch=22,
                       pitch_pad_ind=88, dur_pad_ind=2,
                       pitch_sos_ind=86, pitch_eos_ind=87):
    """
    :param pr_mat: (32, 128) matrix. pr_mat[t, p] indicates a note of pitch p,
    started at time step t, has a duration of pr_mat[t, p] time steps.
    :param max_note_count: the maximum number of notes in a time step,
    including <sos> and <eos> tokens.
    :param max_pitch: the highest pitch in the dataset.
    :param min_pitch: the lowest pitch in the dataset.
    :param pitch_pad_ind: see return value.
    :param dur_pad_ind: see return value.
    :param pitch_sos_ind: sos token.
    :param pitch_eos_ind: eos token.
    :return: pr_mat3d is a (32, max_note_count, 6) matrix. In the last dim,
    the 0th column is for pitch, 1: 6 is for duration in binary repr. Output is
    padded with <sos> and <eos> tokens in the pitch column, but with pad token
    for dur columns.
    """
    pitch_range = max_pitch - min_pitch + 1  # including pad
    pr_mat3d = np.ones((32, max_note_count, 6), dtype=int) * dur_pad_ind
    pr_mat3d[:, :, 0] = pitch_pad_ind
    pr_mat3d[:, 0, 0] = pitch_sos_ind
    cur_idx = np.ones(32, dtype=int)
    for t, p in zip(*np.where(pr_mat != 0)):
        pr_mat3d[t, cur_idx[t], 0] = p - min_pitch
        binary = np.binary_repr(int(pr_mat[t, p]) - 1, width=5)
        pr_mat3d[t, cur_idx[t], 1: 6] = \
            np.fromstring(' '.join(list(binary)), dtype=int, sep=' ')
        if cur_idx[t] == max_note_count - 1:
            continue
        cur_idx[t] += 1
    pr_mat3d[np.arange(0, 32), cur_idx, 0] = pitch_eos_ind
    return pr_mat3d


def expand_chord(chord, shift, relative=False):
    # chord = np.copy(chord)
    root = (chord[0] + shift) % 12
    chroma = np.roll(chord[1: 13], shift)
    bass = (chord[13] + shift) % 12
    root_onehot = np.zeros(12)
    root_onehot[int(root)] = 1
    bass_onehot = np.zeros(12)
    bass_onehot[int(bass)] = 1
    if not relative:
        pass
    #     chroma = np.roll(chroma, int(root))
    # print(chroma)
    # print('----------')
    return np.concatenate([root_onehot, chroma, bass_onehot])


def extract_voicing_chroma_from_pr(pr):
    all_pitches = set(pr[:, 6])
    num_samples = int(((pr[-1][3] + pr[-1][4] / pr[-1][5]) - (pr[0][0] + pr[0][1] / pr[0][2])))
    voicing_chroma = np.zeros((num_samples, 128))
    for pitch in all_pitches:
        voicing_chroma[:, pitch] = 1
    if num_samples < 4:
        voicing_chroma = np.concatenate((voicing_chroma, np.zeros((4 - num_samples, 128))))
    return voicing_chroma


def chord_split(chord, window_size=8, hop_size=8):
    start_downbeat = 0
    end_downbeat = chord.shape[0] // 4
    split_chord = np.empty((0, window_size, 36))
    # print(matrix.shape[0])
    for idx_T in range(start_downbeat * 4, (end_downbeat - (window_size // 4 - 1)) * 4, hop_size):
        if idx_T > chord.shape[0] - 8:
            break
        sample = chord[idx_T:idx_T + window_size, :][np.newaxis, :, :]
        split_chord = np.concatenate((split_chord, sample), axis=0)
    return split_chord


def melody_split(matrix, window_size=32, hop_size=16, vector_size=142):
    start_downbeat = 0
    end_downbeat = matrix.shape[0] // 16
    assert (end_downbeat - start_downbeat >= 2)
    split_matrix = np.empty((0, window_size, vector_size))
    # print(matrix.shape[0])
    # print(matrix.shape[0])
    for idx_T in range(start_downbeat * 16, (end_downbeat - (window_size // 16 - 1)) * 16, hop_size):
        if idx_T > matrix.shape[0] - 32:
            break
        sample = matrix[idx_T:idx_T + window_size, :vector_size][np.newaxis, :, :]
        # print(sample.shape)
        split_matrix = np.concatenate((split_matrix, sample), axis=0)
    return split_matrix


def chord_stretch(c, factor):
    stretched = []
    for i in c:
        for j in range(factor):
            stretched.append(i)
    return np.array(stretched)


def pr_stretch(pr, factor):
    stretched = np.zeros((pr.shape[0] * factor, 128))
    for i in range(len(pr)):
        for pitch in range(128):
            if pr[i][pitch] != 0:
                stretched[i * factor][pitch] = pr[i][pitch] * factor
    return stretched


def accompany_matrix2data(pr_matrix, tempo=120, start_time=0.0, get_list=False):
    alpha = 0.25 * 60 / tempo
    notes = []
    for t in range(pr_matrix.shape[0]):
        for p in range(128):
            if pr_matrix[t, p] >= 1:
                s = alpha * t + start_time
                e = alpha * (t + pr_matrix[t, p]) + start_time
                notes.append(pyd.Note(80, int(p), s, e))
    if get_list:
        return notes
    else:
        acc = pyd.Instrument(program=pyd.instrument_name_to_program('Acoustic Grand Piano'))
        acc.notes = notes
        return acc


def chord_data2matrix(chord_track, downbeats, resolution='beat', chord_expand=True, tolerance=0.125):
    """applicable to triple chords and seventh chords"""
    if resolution == 'beat':
        num_anchors = 4
    elif resolution == 'quarter':
        num_anchors = 16

    NC = [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]
    last_time = 0
    chord_set = []
    chord_time = [[0.0], [0.0]]
    chords_record = []
    # for note in chord_track.notes:
    # handle the 'short notes' (bugs of pretty_midi)
    sorted_chord_track_notes = sorted(chord_track.notes, key=lambda x: x.start)
    for note in sorted_chord_track_notes:
        if note.end - note.start <= 0.1:
            note.end += downbeats[-1] - downbeats[-2]

        if len(chord_set) == 0:
            chord_set.append(note.pitch)
            chord_time[0] = [note.start]
            chord_time[1] = [note.end]
        else:
            if (abs(note.start - np.mean(chord_time[0])) < tolerance) and (
                    abs(note.end - np.mean(chord_time[1])) < tolerance):
                chord_set.append(note.pitch)
                chord_time[0].append(note.start)
                chord_time[1].append(note.end)
            else:
                if last_time < np.mean(chord_time[0]):  # where did we update last_time?
                    chords_record.append({"start": last_time, "end": np.mean(chord_time[0]), "chord": NC})
                chord_set.sort()
                chroma = copy.copy(NC)
                for idx in chord_set:
                    chroma[idx % 12 + 1] = 1
                chroma[0] = chord_set[0] % 12
                chroma[-1] = 0

                # concatenate
                chords_record.append({"start": np.mean(chord_time[0]), "end": np.mean(chord_time[1]), "chord": chroma})
                last_time = np.mean(chord_time[1])
                chord_set = [note.pitch]
                chord_time[0] = [note.start]
                chord_time[1] = [note.end]
    if len(chord_set) > 0:
        if last_time < np.mean(chord_time[0]):
            chords_record.append({"start": last_time, "end": np.mean(chord_time[0]), "chord": NC})
        # chord_set.sort()
        chroma = copy.copy(NC)
        for idx in chord_set:
            chroma[idx % 12 + 1] = 1
        chroma[0] = chord_set[0] % 12
        chroma[-1] = 0
        chords_record.append({"start": np.mean(chord_time[0]), "end": np.mean(chord_time[1]), "chord": chroma})
        last_time = np.mean(chord_time[1])
    chord_table = []
    anchor = 0
    chord = chords_record[anchor]
    start = chord['start']
    downbeats = list(downbeats)
    downbeats.append(downbeats[-1] + (downbeats[-1] - downbeats[-2]))
    for i in range(len(downbeats) - 1):
        s_curr = round(downbeats[i] * 4) / 4
        s_next = round(downbeats[i + 1] * 4) / 4
        delta = (s_next - s_curr) / num_anchors
        for i in range(num_anchors):  # one-beat resolution
            while chord['end'] <= (s_curr + i * delta) and anchor < len(chords_record) - 1:
                anchor += 1
                chord = chords_record[anchor]
                start = chord['start']
            if s_curr + i * delta < start:
                if chord_expand:
                    chord_table.append(expand_chord(chord=NC, shift=0))
                else:
                    chord_table.append(NC)
            else:
                if chord_expand:
                    chord_table.append(expand_chord(chord=chord['chord'], shift=0))
                else:
                    chord_table.append(chord['chord'])
    return np.array(chord_table)


def onset_sus_pr2midi(pr):
    all_notes = []
    i = 0
    for time in range(len(pr)):
        for pitch in range(128):
            if pr[time][pitch] == 2:
                sus = 1
                if time + sus < len(pr):
                    while pr[time + sus][pitch] == 1:
                        sus += 1
                        if time + sus >= len(pr):
                            break
                if i >= 0:
                    all_notes.append(
                        pyd.Note(start=i * 0.125, end=(i + sus) * 0.125, pitch=pitch, velocity=60))
        i += 1
    ins = pyd.Instrument(0)
    ins.notes = all_notes
    midi = pyd.PrettyMIDI()
    midi.instruments.append(ins)
    return midi


def pr2midi(pr):
    all_notes = []
    i = 0
    for time in range(len(pr)):
        for pitch in range(128):
            if pr[time][pitch] != 0:
                all_notes.append(
                    pyd.Note(start=i * 0.125, end=(i + pr[time][pitch]) * 0.125, pitch=pitch, velocity=60))
        i += 1
    ins = pyd.Instrument(0)
    ins.notes = all_notes
    midi = pyd.PrettyMIDI()
    midi.instruments.append(ins)
    return midi


def midi2pr(track, down_sample=1):
    if isinstance(track, pyd.Instrument):
        midi = pyd.PrettyMIDI()
        midi.instruments.append(track)
    elif isinstance(track, pyd.PrettyMIDI):
        midi = track
    else:
        raise Exception
    notes, _, _ = midi_to_source_base(midi)
    max_end = int(max(notes, key=lambda x: x[1])[1]) // down_sample
    pr = np.zeros((max_end, 128))
    for note in notes:
        duration = math.ceil((note[1] - note[0]) / down_sample)
        pr[int(note[0]) // down_sample, note[2]] = duration
    return pr


def midi_to_source_base(midi):
    notes = midi.instruments[0].notes

    # ensure that the first note starts at time 0
    start_time = 1000
    ori_start = 1000
    end_time = 0
    for note in notes:
        if note.start < start_time:
            start_time = note.start
            ori_start = int(round(note.start / 0.125, 0))
    new_notes = []
    for note in notes:
        new_notes.append(pyd.Note(start=note.start - start_time,
                                  end=note.end - start_time,
                                  velocity=note.velocity,
                                  pitch=note.pitch))
    notes = new_notes

    # change note format
    all_formatted_notes = []
    max_end = 0
    for note in notes:
        start = int(round(note.start / 0.125, 0))
        end = int(round(note.end / 0.125, 0))
        if end > max_end:
            max_end = end
        formatted_notes = [start, end, note.pitch, note.velocity]
        all_formatted_notes.append(formatted_notes)
    return all_formatted_notes, ori_start, max_end


def extract_voicing_from_pr(pr: Sequence, chord_length: int, tail: str = 'same'):
    assert tail in ['same', 'cut', 'pad', 'assert_none']
    if tail == 'assert_none':
        assert len(pr) % chord_length == 0
    voicing = []
    chord = [0] * 128
    for time in range(len(pr)):
        if time % chord_length == 0 and time != 0:
            voicing.append(copy.copy(chord))
            for i in range(chord_length - 1):
                voicing.append([0] * 128)
            chord = [0] * 128
        for pitch in range(128):
            if pr[time][pitch] != 0:
                chord[pitch] = chord_length
    if tail == 'cut':
        return voicing
    if tail == 'same':
        if len(pr) % chord_length != 0:
            length = len(pr) % chord_length
            for i in range(len(chord)):
                if chord[i] != 0:
                    chord[i] = length
        voicing.append(copy.copy(chord))
        for i in range(len(pr) % chord_length - 1):
            voicing.append([0] * 128)
        return voicing
    if tail == 'pad' or tail == 'assert_none':
        voicing.append(copy.copy(chord))
        for i in range(chord_length - 1):
            voicing.append([0] * 128)
        return voicing


def pr_to_8d_nmat(pr):
    nmat = []
    for current_time in range(len(pr)):
        for pitch in range(128):
            if pr[current_time][pitch] != 0:
                end = current_time + pr[current_time][pitch]
                note = [current_time // 4, current_time % 4, 4, end // 4, end % 4, 4, pitch, 60]
                nmat.append(note)
    return np.array(nmat)


def nmat_to_pr(nmat):
    max_note = max(nmat, key=lambda i: i[3] * i[5] + i[4])
    max_end = max_note[3] * max_note[5] + max_note[4]
    pr = np.zeros((int(max_end), 128))
    for note in nmat:
        start = note[0] * note[2] + note[1]
        end = note[3] * note[5] + note[4]
        dur = end - start
        pr[start, note[6]] = dur
    return pr


def extract_voicing_from_8d_nmat_2bars(nmat):
    def calc_note_end(weight):
        note_end_unit_count = min_time_unit_count + int(length_unit * weight)
        return note_end_unit_count // min_t[2], note_end_unit_count % min_t[2], min_t[2]

    if not nmat:
        return []

    # calc max end time and min start time
    min_t, max_t = [10000, 0, 4], [0, 0, 4]
    for note in nmat:
        if note[0] + note[1] / note[2] < min_t[0] + min_t[1] / min_t[2]:
            min_t = note[:3]
        if note[3] + note[4] / note[5] > max_t[0] + max_t[1] / max_t[2]:
            max_t = note[3:6]

    # init voicing nmat
    dim0_size = min([14, len(set([note[6] for note in nmat]))])
    new_nmat = np.zeros((dim0_size, 8), dtype=int)

    # calc pitch occurrence for weight
    pitch_count = {}
    for i in range(len([note[6] for note in nmat])):
        if nmat[i][6] in pitch_count.keys():
            pitch_count[nmat[i][6]] += nmat[i][3] * nmat[i][5] + nmat[i][4] - nmat[i][0] * nmat[i][2] + nmat[i][1]
        else:
            pitch_count[nmat[i][6]] = nmat[i][3] * nmat[i][5] + nmat[i][4] - nmat[i][0] * nmat[i][2] + nmat[i][1]
    pitch_max_occur = max(pitch_count.values())
    for i in pitch_count:
        pitch_count[i] = pitch_count[i] / pitch_max_occur

    # calculate voicing
    all_pitches = []
    cursor = 0
    min_time_unit_count = min_t[0] * min_t[2] + min_t[1]
    max_time_unit_count = max_t[0] * max_t[2] + max_t[1]
    length_unit = max_time_unit_count - min_time_unit_count
    for i in range(dim0_size):
        while nmat[cursor][6] in all_pitches:
            cursor += 1
        new_nmat[i][0], new_nmat[i][1], new_nmat[i][2] = min_t[0], min_t[1], min_t[2]
        new_nmat[i][3], new_nmat[i][4], new_nmat[i][5] = calc_note_end(pitch_count[nmat[cursor][6]])
        new_nmat[i][6] = nmat[cursor][6]
        new_nmat[i][7] = nmat[cursor][7]
        all_pitches.append(nmat[cursor][6])
        cursor += 1
    return new_nmat


def extract_voicing_from_8d_nmat(nmat):
    if nmat is None:
        return nmat
    final_nmat = np.zeros((0, 8), dtype=int)
    for i in range(0, int(max(nmat, key=lambda x: x[3])[3]), 4):
        segment_nmat = []
        for note in nmat:
            if i <= note[0] < i + 4:
                segment_nmat.append(note)
        if segment_nmat:
            final_nmat = np.concatenate((final_nmat, extract_voicing_from_8d_nmat_2bars(segment_nmat)))
    return final_nmat


def extract_voicing(midi):
    return pr2midi(nmat_to_pr(extract_voicing_from_8d_nmat(pr_to_8d_nmat(midi2pr(midi)))))


def generate_pop909_test_sample():
    root = r'D:\research\POP909 Phrase Split Data\POP909 Phrase Split Data\Phrase Split Data'
    folder_path = os.path.join(root, random.choice(list(os.listdir(root))))
    file_path = os.path.join(folder_path, random.choice(list(os.listdir(folder_path))))
    while 'midi' in file_path and '8' not in file_path:
        file_path = os.path.join(folder_path, random.choice(list(os.listdir(folder_path))))
    data = np.load(file_path)['piano']
    return pr2midi(data)


def get_whole_song_data(dataset, start_ind, length, shift=0):
    mels = []
    prs = []
    pr_mats = []
    accs = []
    chords = []
    dt_xs = []
    pr_mats_voicing = []
    accs_voicing = []
    voicing_multi_hot = []
    for i in range(start_ind + shift, start_ind + length):
        if (i - start_ind - shift) % 2 != 0:
            continue
        data_sample = dataset[i]
        # mels.append(data_sample['mel_segments'])
        # prs.append(data_sample['prs'])
        pr_mats.append(data_sample['pr_mats'])
        accs.append(data_sample['p_grids'])
        chords.append(data_sample['chord'])
        # dt_xs.append(data_sample['dt_x'])
        # voicing_multi_hot.append(data_sample['voicing_multi_hot'])
        pr_mats_voicing.append(data_sample['pr_mats_voicing'])
        accs_voicing.append(data_sample['p_grids_voicing'])
    # mels = torch.from_numpy(np.array(mels))
    # prs = torch.from_numpy(np.array(prs))
    pr_mats = torch.from_numpy(np.array(pr_mats))
    accs = torch.from_numpy(np.array(accs))
    chords = torch.from_numpy(np.array(chords))
    # dt_xs = torch.from_numpy(np.array(dt_xs))
    # voicing_multi_hot = torch.from_numpy(np.array(voicing_multi_hot))
    pr_mats_voicing = torch.from_numpy(np.array(pr_mats_voicing))
    accs_voicing = torch.from_numpy(np.array(accs_voicing))
    return {
            # 'mel_segments': mels,
            # 'prs': prs,
            'pr_mats': pr_mats,
            'p_grids': accs,
            'pr_mats_voicing': pr_mats_voicing,
            'p_grids_voicing': accs_voicing,
            'chord': chords,
            # 'voicing_multi_hot': voicing_multi_hot,
            # 'dt_x': np.array([])
    }


def get_valid_song_inds(valid_inds, min_bars=16):
    """Inputs 1-d array of inds; returns the start inds of consecutive
    sub-arrays, and length of the array"""
    inds = []
    lengths = []
    length = 0
    for vi, i in enumerate(valid_inds):
        if length == 0:
            start_ind = i
            record_ind = vi
            length = 1
        else:
            if i - start_ind != length:
                if length + 3 >= min_bars:
                    inds.append(record_ind)
                    lengths.append(length)
                start_ind = i
                record_ind = vi
                length = 1
            else:
                length += 1
        if i == valid_inds[-1] and length + 3 >= min_bars:
            inds.append(record_ind)
            lengths.append(length)
    return inds, lengths


if __name__ == '__main__':
    extract_voicing(pm.PrettyMIDI('../experiments/20221017/test4/texture.mid')).write('v.mid')
