import copy
import numpy as np
import pretty_midi as pyd


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


def expand_chord(chord, shift, relative=False):
    root = (chord[0] + shift) % 12
    chroma = np.roll(chord[1: 13], shift)
    bass = (chord[13] + shift) % 12
    root_onehot = np.zeros(12)
    root_onehot[int(root)] = 1
    bass_onehot = np.zeros(12)
    bass_onehot[int(bass)] = 1
    return np.concatenate([root_onehot, chroma, bass_onehot])


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


def pr2midi(pr):
    all_notes = []
    i = 0
    for time in pr:
        for pitch in range(128):
            if time[pitch] != 0:
                all_notes.append(
                    pyd.Note(start=i * 0.125, end=(i + time[pitch]) * 0.125, pitch=pitch, velocity=60))
        i += 1
    ins = pyd.Instrument(0)
    ins.notes = all_notes
    midi = pyd.PrettyMIDI()
    midi.instruments.append(ins)
    return


def midi2pr(track, down_sample=1):
    midi = pyd.PrettyMIDI()
    midi.instruments.append(track)
    notes, _, _ = midi_to_source_base(midi)
    max_end = max(notes, key=lambda x: x[1])[1] // down_sample
    pr = np.zeros((max_end, 128))
    for note in notes:
        duration = (note[1] - note[0]) // down_sample
        pr[(note[0]) // down_sample, note[2]] = duration
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
