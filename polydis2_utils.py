import copy
import os
import random
from typing import Sequence

import numpy as np
import pretty_midi

from format_converter import midi2pr, pr2midi


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


def extract_voicing_from_8d_nmat(nmat):
    if nmat is None:
        return None
    min_t, max_t = [10000, 0, 4], [0, 0, 4]
    for note in nmat:
        if note[0] + note[1] / note[2] < min_t[0] + min_t[1] / min_t[2]:
            min_t = note[:3]
        if note[3] + note[4] / note[5] > max_t[0] + max_t[1] / max_t[2]:
            max_t = note[3:6]
    dim0_size = min([14, len(set([note[6] for note in nmat]))])
    new_nmat = np.zeros((dim0_size, 8), dtype=int)
    all_pitches = []
    cursor = 0
    for i in range(dim0_size):
        while nmat[cursor][6] in all_pitches:
            cursor += 1
        new_nmat[i][0], new_nmat[i][1], new_nmat[i][2] = min_t[0], min_t[1], min_t[2]
        new_nmat[i][3], new_nmat[i][4], new_nmat[i][5] = max_t[0], max_t[1], max_t[2]
        new_nmat[i][6] = nmat[cursor][6]
        new_nmat[i][7] = nmat[cursor][7]
        all_pitches.append(nmat[cursor][6])
        cursor += 1
    return new_nmat


def generate_pop909_test_sample():
    root = r'D:\research\POP909 Phrase Split Data\POP909 Phrase Split Data\Phrase Split Data'
    folder_path = os.path.join(root, random.choice(list(os.listdir(root))))
    file_path = os.path.join(folder_path, random.choice(list(os.listdir(folder_path))))
    while 'midi' in file_path and '8' not in file_path:
        file_path = os.path.join(folder_path, random.choice(list(os.listdir(folder_path))))
    data = np.load(file_path)['piano']
    return pr2midi(data)


if __name__ == '__main__':
    MIDI_IN_PATH = 'test.mid'
    MIDI_OUT_PATH = 'voicing.mid'
    pr = midi2pr(pretty_midi.PrettyMIDI(MIDI_IN_PATH))
    voicing = extract_voicing_from_pr(pr, 16)
    pr2midi(voicing).write(MIDI_OUT_PATH)
