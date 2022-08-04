import copy

import numpy as np
import pretty_midi
import torch
import pretty_midi as pyd
from model import DisentangleVAE
from ptvae import PtvaeDecoder


def chordSplit(chord, WINDOWSIZE=8, HOPSIZE=8):
    start_downbeat = 0
    end_downbeat = chord.shape[0] // 4
    splittedChord = np.empty((0, WINDOWSIZE, 36))
    # print(matrix.shape[0])
    for idx_T in range(start_downbeat * 4, (end_downbeat - (WINDOWSIZE // 4 - 1)) * 4, HOPSIZE):
        if idx_T > chord.shape[0] - 8:
            break
        sample = chord[idx_T:idx_T + WINDOWSIZE, :][np.newaxis, :, :]
        splittedChord = np.concatenate((splittedChord, sample), axis=0)
    return splittedChord


def melodySplit(matrix, WINDOWSIZE=32, HOPSIZE=16, VECTORSIZE=142):
    start_downbeat = 0
    end_downbeat = matrix.shape[0] // 16
    assert (end_downbeat - start_downbeat >= 2)
    splittedMatrix = np.empty((0, WINDOWSIZE, VECTORSIZE))
    # print(matrix.shape[0])
    # print(matrix.shape[0])
    for idx_T in range(start_downbeat * 16, (end_downbeat - (WINDOWSIZE // 16 - 1)) * 16, HOPSIZE):
        if idx_T > matrix.shape[0] - 32:
            break
        sample = matrix[idx_T:idx_T + WINDOWSIZE, :VECTORSIZE][np.newaxis, :, :]
        # print(sample.shape)
        splittedMatrix = np.concatenate((splittedMatrix, sample), axis=0)
    return splittedMatrix


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


def accomapnimentGeneration(pr_matrix, tempo=120):
    # print(piano_roll.shape, type(piano_roll))
    pt_decoder = PtvaeDecoder(note_embedding=None, dec_dur_hid_size=64, z_size=512)
    start = 0
    tempo = tempo
    midiReGen = pyd.PrettyMIDI(initial_tempo=tempo)
    texture_track = pyd.Instrument(program=pyd.instrument_name_to_program('Acoustic Grand Piano'))
    for idx in range(0, pr_matrix.shape[0]):
        if pr_matrix.shape[-1] == 6:
            pr, _ = pt_decoder.grid_to_pr_and_notes(grid=pr_matrix[idx], bpm=tempo, start=0)
        else:
            pr = pr_matrix[idx]
        # print(pr.shape)
        texture_notes = accompany_matrix2data(pr_matrix=pr, tempo=tempo, start_time=start, get_list=True)
        texture_track.notes += texture_notes
        start += 60 / tempo * 8
    midiReGen.instruments.append(texture_track)
    return midiReGen


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


def chord_data2matrix(chord_track, downbeats, resolution='beat', chord_expand=True, tolerence=0.125):
    """applicable to triple chords and seventh chords"""
    if resolution == 'beat':
        num_anchords = 4
    elif resolution == 'quater':
        num_anchords = 16

    NC = [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]
    last_time = 0
    chord_set = []
    chord_time = [[0.0], [0.0]]
    chordsRecord = []
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
            if (abs(note.start - np.mean(chord_time[0])) < tolerence) and (
                    abs(note.end - np.mean(chord_time[1])) < tolerence):
                chord_set.append(note.pitch)
                chord_time[0].append(note.start)
                chord_time[1].append(note.end)
            else:
                if last_time < np.mean(chord_time[0]):  # where did we update last_time?
                    chordsRecord.append({"start": last_time, "end": np.mean(chord_time[0]), "chord": NC})
                chord_set.sort()
                chroma = copy.copy(NC)
                for idx in chord_set:
                    chroma[idx % 12 + 1] = 1
                chroma[0] = chord_set[0] % 12
                chroma[-1] = 0

                # concatenate
                chordsRecord.append({"start": np.mean(chord_time[0]), "end": np.mean(chord_time[1]), "chord": chroma})
                last_time = np.mean(chord_time[1])
                chord_set = []
                chord_set.append(note.pitch)
                chord_time[0] = [note.start]
                chord_time[1] = [note.end]
    if len(chord_set) > 0:
        if last_time < np.mean(chord_time[0]):
            chordsRecord.append({"start": last_time, "end": np.mean(chord_time[0]), "chord": NC})
        # chord_set.sort()
        chroma = copy.copy(NC)
        for idx in chord_set:
            chroma[idx % 12 + 1] = 1
        chroma[0] = chord_set[0] % 12
        chroma[-1] = 0
        chordsRecord.append({"start": np.mean(chord_time[0]), "end": np.mean(chord_time[1]), "chord": chroma})
        last_time = np.mean(chord_time[1])
    print(chordsRecord)
    ChordTable = []
    anchor = 0
    chord = chordsRecord[anchor]
    start = chord['start']
    downbeats = list(downbeats)
    downbeats.append(downbeats[-1] + (downbeats[-1] - downbeats[-2]))
    for i in range(len(downbeats) - 1):
        s_curr = round(downbeats[i] * 4) / 4
        s_next = round(downbeats[i + 1] * 4) / 4
        delta = (s_next - s_curr) / num_anchords
        for i in range(num_anchords):  # one-beat resolution
            while chord['end'] <= (s_curr + i * delta) and anchor < len(chordsRecord) - 1:
                anchor += 1
                chord = chordsRecord[anchor]
                start = chord['start']
            if s_curr + i * delta < start:
                if chord_expand:
                    ChordTable.append(expand_chord(chord=NC, shift=0))
                else:
                    ChordTable.append(NC)
            else:
                if chord_expand:
                    ChordTable.append(expand_chord(chord=chord['chord'], shift=0))
                else:
                    ChordTable.append(chord['chord'])
    return np.array(ChordTable)


def inference(chord_table, acc_emsemble):
    acc_emsemble = melodySplit(acc_emsemble, WINDOWSIZE=32, HOPSIZE=32, VECTORSIZE=128)
    chord_table = chordSplit(chord_table, 8, 8)
    if torch.cuda.is_available():
        model = DisentangleVAE.init_model(torch.device('cuda')).cuda()
        checkpoint = torch.load('data/model_master_final.pt')
        model.load_state_dict(checkpoint)
        pr_matrix = torch.from_numpy(acc_emsemble).float().cuda()
        # pr_matrix_shifted = torch.from_numpy(pr_matrix_shifted).float().cuda()
        gt_chord = torch.from_numpy(chord_table).float().cuda()
        # print(gt_chord.shape, pr_matrix.shape)
        est_x = model.inference(pr_matrix, gt_chord, sample=False)
        # print('est:', est_x.shape)
        # est_x_shifted = model.inference(pr_matrix_shifted, gt_chord, sample=False)
        midiReGen = accomapnimentGeneration(est_x, 120)
        return midiReGen
        # midiReGen.write('accompaniment_test_NEW.mid')
    else:
        model = DisentangleVAE.init_model(torch.device('cpu'))
        checkpoint = torch.load('data/model_master_final.pt', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        pr_matrix = torch.from_numpy(acc_emsemble).float()
        gt_chord = torch.from_numpy(chord_table).float()
        est_x = model.inference(pr_matrix, gt_chord, sample=False)
        midiReGen = accomapnimentGeneration(est_x, 120)
        return midiReGen


def midi2pr(track):
    # return track.get_piano_roll(fs=120)

    pianoroll = np.zeros((128, 128))
    if not track.notes:
        return pianoroll

    midi = pyd.PrettyMIDI()
    midi.instruments.append(track)
    fs = 8  # dataset independent
    # Returns matrix of shape (128, time) with summed velocities.
    pr = midi.get_piano_roll(fs=fs)
    pr = np.where(pr > 0, 1, 0)
    pr = pr.T
    pr = pr[:, 0: 128]
    return pr


def pr2midi(pr):
    all_notes = []
    i = 0
    for time in pr:
        for pitch in range(128):
            if time[pitch] != 0:
                all_notes.append(
                    pretty_midi.Note(start=i * 0.125, end=(i + time[pitch]) * 0.125, pitch=pitch, velocity=60))
        i += 1
    ins = pretty_midi.Instrument(0)
    ins.notes = all_notes
    midi = pretty_midi.PrettyMIDI()
    midi.instruments.append(ins)
    return midi


if __name__ == '__main__':
    path = 'test.mid'
    midi = pyd.PrettyMIDI(path)
    # print(midi.get_piano_roll(fs=120))
    chord_table = chord_data2matrix(midi.instruments[0], midi.get_downbeats(), 'quater')
    chord_table = chord_table[::4, :]
    acc_emsemble = midi2pr(midi.instruments[0])
    gen = inference(chord_table, acc_emsemble)
    gen.write('gen.mid')
