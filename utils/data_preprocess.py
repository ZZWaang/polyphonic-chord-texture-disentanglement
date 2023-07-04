import os
import numpy as np
import pretty_midi
from tqdm import tqdm

from utils import chord_data2matrix, midi2pr

np.set_printoptions(threshold=np.inf)


def prepare_pop909_stage_a_dataset():
    all_chord_data = []
    all_prs = []
    song_idx = 0
    song_prs = []
    song_chord_data = []
    for file in tqdm(os.listdir('../zv_new')):
        check = int(file.split('-')[0])
        if check != song_idx:
            song_idx = check
            if len(song_prs) != 0:
                all_prs.append(song_prs)
                all_chord_data.append(song_chord_data)
                song_prs = []
                song_chord_data = []
            else:
                print(file)
        midi = pretty_midi.PrettyMIDI('../zv_new/' + file)
        downbeats = midi.get_downbeats()
        track = midi.instruments[0]
        chord_data = chord_data2matrix(track, downbeats, 'quarter', False)
        chord_data = chord_data[::2]
        for i in chord_data:
            my_sum = sum(i[1:-1])
            if my_sum >= 8:
                continue
        pr = midi2pr(track)
        pr = pr[::2]
        if len(pr) != 64 or len(chord_data) != 64:
            continue
        song_prs.append(pr)
        song_chord_data.append(chord_data)

    print(len(all_prs), len(all_chord_data))
    all_prs = np.array(all_prs)
    all_chord_data = np.array(all_chord_data)
    np.savez_compressed('../data/pop909_stage_a.npz', pr=all_prs, c=all_chord_data)


def preprocess_zv():
    for file in os.listdir('../zv'):
        midi = pretty_midi.PrettyMIDI('../zv/' + file)
        track = midi.instruments[0]
        new_track = pretty_midi.Instrument(program=0)
        for note in track.notes:
            new_track.notes.append(
                pretty_midi.Note(start=note.start // 2 * 2, end=note.start // 2 * 2 + 2, pitch=note.pitch,
                                 velocity=note.velocity))
        midi.instruments = [new_track]
        midi.write('../zv_new/' + file)


if __name__ == '__main__':
    prepare_pop909_stage_a_dataset()
