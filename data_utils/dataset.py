import numpy as np
from torch.utils.data import Dataset
import glob
import os
import pandas as pd
from tqdm import tqdm
from data_utils.score import PolyphonicMusic, NikoChordProgression
from torch.utils.data import DataLoader
from utils.utils import ext_nmat_to_pr, ext_nmat_to_mel_pr, \
    augment_pr, augment_mel_pr, pr_to_onehot_pr, piano_roll_to_target, \
    target_to_3dtarget, expand_chord

DATA_PATH = os.path.join('../data', 'POP09-PIANOROLL-4-bin-quantization')
INDEX_FILE_PATH = os.path.join('../data', 'index.xlsx')
SEED = 3345


class ArrangementDataset(Dataset):

    def __init__(self, data, indicator, shift_low, shift_high, num_bar=8,
                 ts=4, contain_chord=False, contain_voicing=False):
        super(ArrangementDataset, self).__init__()
        self.data = data
        self.indicator = indicator
        self.shift_low = shift_low
        self.shift_high = shift_high
        self.num_sample = int(self.indicator.sum())
        self.valid_inds = self._get_sample_inds()
        self.num_bar = num_bar
        self.ts = ts
        self.contain_chord = contain_chord
        self.contain_voicing = contain_voicing
        self.cache = {}

    def _get_sample_inds(self):
        valid_inds = []
        for i, ind in enumerate(self.indicator):
            if ind:
                valid_inds.append(i)
        return valid_inds

    @staticmethod
    def _translate(track, translation):
        if track is None:
            return track
        track = np.copy(track)
        track[:, 0] -= translation
        track[:, 3] -= translation
        return track

    def _combine_segments(self, segment):
        first, second = segment
        if first is None and second is None:
            nmat = None
        elif first is None:
            nmat = ArrangementDataset._translate(second, -self.ts)
        elif second is None:
            nmat = first
        else:
            second = ArrangementDataset._translate(second, -self.ts)
            nmat = np.concatenate([first, second], axis=0)

        return nmat

    def __len__(self):
        # consider data augmentation here
        return self.num_sample * (self.shift_high - self.shift_low + 1)

    def __getitem__(self, id):
        if id in self.cache:
            return self.cache[id]
        # separate id into (no, shift) pair
        no = id // (self.shift_high - self.shift_low + 1)
        shift = id % (self.shift_high - self.shift_low + 1) + self.shift_low

        ind = self.valid_inds[no]
        data = self.data[ind: ind + self.num_bar]
        # data is a list of length num_bar
        # each element is a list containing mel and acc matrices

        mel = [x[0] for x in data]
        mel_segments = \
            [ext_nmat_to_mel_pr(self._combine_segments(mel[i: i + 2]))
             for i in range(0, self.num_bar, 2)]

        # consider when there are None type segments.
        # translate, add together and change format.
        # break into four segments

        acc = [x[1] for x in data]
        acc_segments = [ext_nmat_to_pr(self._combine_segments(acc[i: i + 2]))
                        for i in range(0, self.num_bar, 2)]

        # do augmentation
        mel_segments = np.array([augment_mel_pr(pr, shift) for pr in mel_segments])
        acc_segments = np.array([augment_pr(pr, shift) for pr in acc_segments])

        # deal with the polyphonic ones
        prs = np.array([pr_to_onehot_pr(pr) for pr in acc_segments])
        pr_mats = np.array([piano_roll_to_target(pr) for pr in prs])
        p_grids = np.array([target_to_3dtarget(pr_mat,
                                               max_note_count=16,
                                               max_pitch=128,
                                               min_pitch=0,
                                               pitch_pad_ind=130,
                                               pitch_sos_ind=128,
                                               pitch_eos_ind=129)
                            for pr_mat in pr_mats])
        # for this task
        prs = prs[0]
        pr_mats = pr_mats[0]
        p_grids = p_grids[0]

        pr_mats_voicing = None
        p_grids_voicing = None
        voicing_multi_hot = None

        if self.contain_voicing:
            voicing = [x[3] for x in data]
            voicing_segments = [ext_nmat_to_pr(self._combine_segments(voicing[i: i + 2]))
                                for i in range(0, self.num_bar, 2)]
            voicing_segments = np.array([augment_pr(pr, shift) for pr in voicing_segments])
            prs_voicing = np.array([pr_to_onehot_pr(pr) for pr in voicing_segments])
            pr_mats_voicing = np.array([piano_roll_to_target(pr) for pr in prs_voicing])
            p_grids_voicing = np.array([target_to_3dtarget(pr_mat_voicing,
                                                           max_note_count=16,
                                                           max_pitch=128,
                                                           min_pitch=0,
                                                           pitch_pad_ind=130,
                                                           pitch_sos_ind=128,
                                                           pitch_eos_ind=129)
                                        for pr_mat_voicing in pr_mats_voicing])
            # for this task
            pr_mats_voicing = pr_mats_voicing[0]
            p_grids_voicing = p_grids_voicing[0]
            bar1_multi_hot = np.array([np.logical_or(pr_mats_voicing[0], np.zeros(128))], dtype=int).repeat(16, axis=0)
            bar2_multi_hot = np.array([np.logical_or(pr_mats_voicing[16], np.zeros(128))], dtype=int).repeat(16, axis=0)
            voicing_multi_hot = np.concatenate((bar1_multi_hot, bar2_multi_hot), axis=0)

        if self.contain_chord:
            chord = [x[2] for x in data]
            chord = np.concatenate(chord, axis=0)
            chord = np.array([expand_chord(c, shift) for c in chord])
        else:
            chord = None

        batch_data = {'mel_segments': mel_segments,
                      'prs': prs,
                      'pr_mats': pr_mats,
                      'p_grids': p_grids,
                      'chord': chord,
                      'dt_x': np.array([]),
                      'pr_mats_voicing': pr_mats_voicing,
                      'p_grids_voicing': p_grids_voicing,
                      'voicing_multi_hot': voicing_multi_hot}
        self.cache[id] = batch_data
        return batch_data


def detrend_pianotree(piano_tree, c):
    # piano_tree: (32, 16, 6)
    root = np.argmax(c[:, 0: 12], axis=-1)
    bass = np.argmax(c[:, 24:], axis=-1)
    # print(root)
    dur = piano_tree[:, :, 1:].reshape((8, 4, 16, 5))
    pitch = piano_tree[:, :, 0]  # (32, 16)
    pitch = pitch.reshape(8, 4, 16)
    # octave = pitch // 12
    # degree = (pitch.reshape((8, 4, 16)) - root) % 12  # (8, 4, 16)

    map_dic = {(1, 0): 0, (0, 1): 1, (0, 0): 2, (1, 1): 3}
    deg_table = [0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6]
    semi_table = [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1]
    chroma = np.array([np.roll(cc, shift=-rr, axis=-1)
                       for cc, rr in zip(c[:, 12: 24], root)])
    chroma_states = get_chroma_state(chroma, map_dic)  # (8, 7)

    is_notes = np.zeros((8, 4, 16, 4), dtype=int)
    is_basses = np.zeros((8, 4, 16, 3), dtype=int)
    octaves = np.zeros((8, 4, 16, 12), dtype=int)
    degs = np.zeros((8, 4, 16, 8), dtype=int)
    n_states = np.zeros((8, 4, 16, 7), dtype=int)
    for t in range(8):
        chroma_state = chroma_states[t]
        rr = root[t]
        bb = bass[t]
        has_bass = False
        for i in range(4):
            for j in range(16):
                is_note, is_bass, octave, scale_deg, n_state = \
                    convert_note(pitch[t, i, j], chroma_state, rr, bb,
                                 deg_table, semi_table)
                if has_bass:
                    is_bass = 0
                else:
                    has_bass = True
                is_notes[t, i, j, is_note] = 1
                is_basses[t, i, j, is_bass] = 1
                octaves[t, i, j, octave] = 1
                degs[t, i, j, scale_deg] = 1
                n_states[t, i, j, n_state] = 1
    notes = np.concatenate([is_notes, is_basses, octaves, degs, n_states, dur],
                           axis=-1)
    notes = notes.reshape((32, 16, -1))
    return notes


def get_chroma_state(chroma, map_dic):
    chroma_states = np.zeros((8, 7), dtype=int)
    chroma_states[:, [0, 4]] = ((1 - chroma[:, [0, 7]]) * 2).astype(int)
    chroma_states[:, 1] = np.array([map_dic[tuple(cc)]
                                    for cc in chroma[:, [1, 2]]], dtype=int)
    chroma_states[:, 2] = np.array([map_dic[tuple(cc)]
                                    for cc in chroma[:, [3, 4]]], dtype=int)
    chroma_states[:, 3] = np.array([map_dic[tuple(cc)]
                                    for cc in chroma[:, [5, 6]]], dtype=int)
    chroma_states[:, 5] = np.array([map_dic[tuple(cc)]
                                    for cc in chroma[:, [8, 9]]], dtype=int)
    chroma_states[:, 6] = np.array([map_dic[tuple(cc)]
                                    for cc in chroma[:, [10, 11]]], dtype=int)
    return chroma_states


def convert_note(pitch, chroma_state, root, bass, deg_table, semi_table):
    # chroma state: length 7
    # chroma_state: 0: has_low or only one, 1: has high,
    # 2: neither or has only one; 3: all
    # note: 0 - 11
    if pitch == 128:
        return 1, 2, 11, 7, 6
    elif pitch == 129:
        return 2, 2, 11, 7, 6
    elif pitch == 130:
        return 3, 2, 11, 7, 6

    octave = pitch // 12
    degree = (pitch - root) % 12
    is_bass = 1 if bass == degree else 0
    scale_deg = deg_table[degree]  # 0 - 7
    c_state = chroma_state[scale_deg]  # 0 - 4
    semitone = semi_table[scale_deg]  # 0 - 2
    if c_state == 0:
        n_state = 0 if semitone else 1
    elif c_state == 1:
        n_state = 1 if semitone else 0
    elif c_state == 2:
        n_state = semitone + 2
    elif c_state == 3:
        n_state = semitone + 4
    else:
        raise NotImplementedError
    return 0, is_bass, octave, scale_deg, n_state


def collect_data_fns():
    valid_files = []
    files = glob.glob(os.path.join(DATA_PATH, '*.npz'))
    print('The folder contains %d .npz files.' % len(files))
    df = pd.read_excel(INDEX_FILE_PATH)
    for file in files:
        song_id = file.split('/')[-1][0: 3]
        meta_data = df[df.song_id == int(song_id)]
        num_beats = meta_data.num_beats_per_measure.values[0]
        if int(num_beats) == 2:
            valid_files.append(file)
    print('Selected %d files, all are in duple meter.' % len(valid_files))
    return valid_files


def init_music(fn, prepare_voicing=False):
    data = np.load(fn)
    beat = data['beat']
    chord = data['chord']
    melody = data['melody']
    bridge = data['bridge']
    piano = data['piano']
    music = PolyphonicMusic([melody, bridge, piano], beat, chord, [70, 0, 0], prepare_voicing=prepare_voicing)
    return music


def split_dataset(length, portion):
    train_ind = np.random.choice(length, int(length * (portion - 1) / (portion + 1)),
                                 replace=False)
    valid_n_test_ind = np.setdiff1d(np.arange(0, length), train_ind)
    valid_ind = np.random.choice(valid_n_test_ind, len(valid_n_test_ind) // 2, replace=False)
    test_ind = np.setdiff1d(valid_n_test_ind, valid_ind)
    return train_ind, valid_ind, test_ind


def wrap_dataset(fns, ids, shift_low, shift_high, num_bar=8, niko=False, prepare_voicing=False, cache_name=''):
    def load_cache():
        if 'cache' in os.listdir('./'):
            if f'wrap_dataset_cache_{cache_name}.npz' in os.listdir('cache'):
                cache = np.load(f'cache/wrap_dataset_cache_{cache_name}.npz', allow_pickle=True)
                return cache['data'], cache['indicator']
            else:
                return [], []
        else:
            return [], []

    def save_cache():
        os.makedirs('cache', exist_ok=True)
        np.savez_compressed(f'cache/wrap_dataset_cache_{cache_name}', data=data, indicator=indicator)

    data, indicator = load_cache()
    if data != []:
        print(f'Using cached dataset with cache name {cache_name}')
        dataset = ArrangementDataset(data, indicator, shift_low, shift_high,
                                     num_bar=num_bar, contain_chord=True, contain_voicing=prepare_voicing)
        return dataset
    if niko:
        pr, c = fns['pr'], fns['c']
    for ind in tqdm(ids):
        music = init_music(fns[ind], prepare_voicing=prepare_voicing) if not niko else NikoChordProgression(pr[ind],
                                                                                                            c[ind])
        data_track, indct, db_pos = music.prepare_data(num_bar=num_bar)
        data += data_track
        indicator.append(indct)
    indicator = np.concatenate(indicator)
    dataset = ArrangementDataset(data, indicator, shift_low, shift_high,
                                 num_bar=num_bar, contain_chord=True, contain_voicing=prepare_voicing)
    save_cache()
    return dataset


def prepare_dataset(seed, bs_train, bs_val, portion=8, shift_low=-6, shift_high=5,
                    num_bar=2, random_train=True, random_val=False):
    # fns = collect_data_fns()
    import pickle
    with open('data/ind.pkl', 'rb') as f:
        fns = pickle.load(f)
    np.random.seed(seed)
    train_ids, val_ids, test_ids = split_dataset(len(fns), portion)
    train_set = wrap_dataset(fns, train_ids, shift_low, shift_high, num_bar=num_bar, cache_name='pop909_train')
    val_set = wrap_dataset(fns, val_ids, 0, 0, num_bar=num_bar, cache_name='pop909_val')
    print(len(train_set), len(val_set))
    train_loader = DataLoader(train_set, bs_train, random_train)
    val_loader = DataLoader(val_set, bs_val, random_val)
    return train_loader, val_loader


def prepare_dataset_pop909_voicing(seed, bs_train, bs_val, portion=8, shift_low=-6, shift_high=5,
                                   num_bar=2, random_train=True, random_val=False):
    # fns = collect_data_fns()
    print('Loading Training Data...')
    import pickle
    with open('data/ind.pkl', 'rb') as f:
        fns = pickle.load(f)
    np.random.seed(seed)
    train_ids, val_ids, test_ids = split_dataset(len(fns), portion)
    print('Constructing Training Set')
    train_set = wrap_dataset(fns, train_ids, shift_low, shift_high, num_bar=num_bar, prepare_voicing=True,
                             cache_name='pop909_voicing_train')
    print('Constructing Validation Set')
    val_set = wrap_dataset(fns, val_ids, 0, 0, num_bar=num_bar, prepare_voicing=True, cache_name='pop909_voicing_val')
    print(f'Done with {len(train_set)} training samples, {len(val_set)} validation samples')
    train_loader = DataLoader(train_set, bs_train, random_train)
    val_loader = DataLoader(val_set, bs_val, random_val)
    return train_loader, val_loader


def prepare_dataset_niko(seed, bs_train, bs_val,
                         portion=8, shift_low=-6, shift_high=5, num_bar=2, random_train=True, random_val=False):
    """
    Return the dataloaders of the niko dataset
    """
    print('Loading Training Data...')
    data = np.load('../data/poly-dis-niko.npz', allow_pickle=True)
    np.random.seed(seed)
    train_ids, val_ids, test_ids = split_dataset(len(data['pr']), portion)
    np.save('test_ids.npy', test_ids)
    print('Constructing Training Set')
    train_set = wrap_dataset(data, train_ids, shift_low, shift_high,
                             num_bar=num_bar, niko=True, cache_name='niko_train')
    print('Constructing Validation Set')
    val_set = wrap_dataset(data, val_ids, 0, 0, num_bar=num_bar, niko=True, cache_name='niko_val')
    print(f'Done with {len(train_set)} training samples, {len(val_set)} validation samples')
    train_loader = DataLoader(train_set, bs_train, random_train)
    val_loader = DataLoader(val_set, bs_val, random_val)
    return train_loader, val_loader


if __name__ == '__main__':
    tl, vl = prepare_dataset(SEED, 32, 32)

    print(len(tl))
    print(len(vl))
