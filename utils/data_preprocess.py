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
    flattened_chord_data = []
    flattened_prs = []
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
                pass
        midi = pretty_midi.PrettyMIDI('../zv_new/' + file)
        downbeats = midi.get_downbeats()
        track = midi.instruments[0]
        chord_data = chord_data2matrix(track, downbeats, 'quarter', False)
        chord_data = chord_data[::2]
        for i in chord_data:
            my_sum = sum(i[1:-1])
            if my_sum >= 8:
                continue
        pr = midi2pr(track, down_sample=2)
        if pr.shape != (64, 128) or chord_data.shape != (64, 14):
            continue
        song_prs.append(pr)
        song_chord_data.append(chord_data)
        flattened_prs.append(pr)
        flattened_chord_data.append(chord_data)

    all_prs = np.array(all_prs)
    all_chord_data = np.array(all_chord_data)
    flattened_prs = np.array(flattened_prs)
    flattened_chord_data = np.array(flattened_chord_data)
    print(all_prs.shape, all_chord_data.shape)
    print(flattened_prs.shape, flattened_chord_data.shape)
    np.savez_compressed('../data/pop909_stage_a_no_full_song_fixed.npz', pr=flattened_prs, c=flattened_chord_data)


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
    # file = np.load('../data/pop909_stage_a.npz', allow_pickle=True)
    # data = file['pr']
    # c = file['c']
    # for i in data:
    #     for j in i:
    #         for t in j:
    #             count = 0
    #             for p in range(128):
    #                 if t[p] != 0:
    #                     t[p] = 8
    # # for i in data:
    # #     for j in i:
    # #         for t in j:
    # #             count = 0
    # #             for p in range(128):
    # #                 if t[p] != 0:
    # #                     print(t[p])
    # np.savez_compressed('../data/pop909_stage_a.npz', pr=data, c=c, allow_pickle=True)
