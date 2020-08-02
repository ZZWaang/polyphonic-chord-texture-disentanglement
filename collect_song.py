from dataset_loaders import MusicDataLoaders
from dataset import SEED
import numpy as np
import torch


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


def get_whole_song_data(dataset, start_ind, length, shift=0):
    mels = []
    prs = []
    pr_mats = []
    accs = []
    chords = []
    dt_xs = []
    for i in range(start_ind + shift, start_ind + length):
        if (i - start_ind - shift) % 2 != 0:
            continue
        mel_segments, pr, pr_mat, p_grids, chord, dt_x = dataset[i]
        mels.append(mel_segments)
        prs.append(pr)
        pr_mats.append(pr_mat)
        accs.append(p_grids)
        chords.append(chord)
        dt_xs.append(dt_x)
    mels = torch.from_numpy(np.array(mels))
    pr = torch.from_numpy(np.array(pr))
    pr_mats = torch.from_numpy(np.array(pr_mats))
    accs = torch.from_numpy(np.array(accs))
    chords = torch.from_numpy(np.array(chords))
    dt_xs = torch.from_numpy(np.array(dt_xs))
    return mels, pr, pr_mats, accs, chords, dt_xs


class SongDataset:

    def __init__(self, dataset):
        self.dataset = dataset
        song_ind, song_len = get_valid_song_inds(dataset.valid_inds,
                                                 min_bars=16)
        self.song_ind = song_ind
        self.song_len = song_len

    def get_song_batch(self, song_id, length=None, shift=0):
        if length is None:
            length = self.song_len[song_id]
        assert length + shift <= self.song_len[song_id]
        batch = get_whole_song_data(self.dataset, self.song_ind[song_id],
                                    length + shift, shift)
        return batch


class SongDatasets:

    def __init__(self, train_dataset, val_dataset):
        self.song_dataset_t = SongDataset(train_dataset)
        self.song_dataset_v = SongDataset(val_dataset)

    def get_song_batch(self, dataset_id, song_id, length, shift):
        if dataset_id == 0:
            dataset = self.song_dataset_t
        else:
            dataset = self.song_dataset_v
        batch = dataset.get_song_batch(song_id, length, shift)
        return batch

    def valid_length(self, dataset_id, song_id,  length):
        if length is not None:
            return length
        if dataset_id == 0:
            dataset = self.song_dataset_t
        else:
            dataset = self.song_dataset_v
        return dataset.song_len[song_id]

    def get_msg(self, dataset_id, song_id, length, shift):
        if dataset_id == 0:
            dataset = self.song_dataset_t
        else:
            dataset = self.song_dataset_v
        if length is None:
            length = dataset.song_len[song_id]
        return '_'.join([str(dataset_id), str(song_id),
                         str(length), str(shift)])


if __name__ == '__main__':
    batch_size = 32
    data_loaders = \
        MusicDataLoaders.get_loaders(SEED, bs_train=batch_size,
                                     bs_val=batch_size,
                                     portion=8, shift_low=-6, shift_high=5,
                                     num_bar=2,
                                     contain_chord=True, random_train=False,
                                     random_val=False)
    train_loader = data_loaders.train_loader
    val_loader = data_loaders.val_loader

    val_dataset = val_loader.dataset
    print(len(val_dataset.data), val_dataset.indicator.shape)
    inds, lengths = get_valid_song_inds(val_dataset.valid_inds)

    # print(inds[0: 10], lengths[0: 10])
    for i, (ind, length) in enumerate(zip(inds, lengths)):
        batch = get_whole_song_data(val_dataset, ind, length, shift=0)
        for b in batch:
            print(b.size())
        if i == 2:
            break

