import numpy as np
import pretty_midi
import torch

from data_utils.dataset import prepare_dataset_niko, prepare_dataset, prepare_dataset_pop909_voicing, \
    prepare_dataset_pop909_stage_a, prepare_dataset_pop909_xuran
from amc_dl.torch_plus import DataLoaders
from amc_dl.torch_plus import TrainingInterface


class MusicDataLoaders(DataLoaders):

    @staticmethod
    def get_loaders(seed, dataset_name, bs_train, bs_val,
                    portion=8, shift_low=-6, shift_high=5, num_bar=2,
                    contain_chord=True, random_train=True, random_val=False, full_song=False):
        if dataset_name == 'niko':
            train, val = prepare_dataset_niko(seed, bs_train, bs_val, portion, shift_low,
                                              shift_high, num_bar, random_train, random_val)
        elif dataset_name == 'pop909_voicing':
            train, val = prepare_dataset_pop909_voicing(seed, bs_train, bs_val, portion, shift_low,
                                                        shift_high, num_bar, random_train, random_val, full_song)
        elif dataset_name == 'pop909':
            train, val = prepare_dataset(seed, bs_train, bs_val, portion, shift_low,
                                         shift_high, num_bar, random_train, random_val)
        elif dataset_name == 'pop909_stage_a':
            train, val = prepare_dataset_pop909_stage_a(seed, bs_train, bs_val, portion, shift_low,
                                                        shift_high, num_bar, random_train, random_val, full_song)
        elif dataset_name == 'pop909_xuran':
            train, val = prepare_dataset_pop909_xuran(seed, bs_train, bs_val, portion, shift_low,
                                                      shift_high, num_bar, random_train, random_val, full_song)
        else:
            raise Exception
        return MusicDataLoaders(train, val, bs_train, bs_val)


def save_midi(pr, name):
    midi = pretty_midi.PrettyMIDI()
    track = pretty_midi.Instrument(program=0)
    for i in range(32):
        for j in range(128):
            if pr[i][j] != 0:
                track.notes.append(pretty_midi.Note(start=i * 0.125, end=(i + 1) * 0.125, pitch=j, velocity=60))
    midi.instruments.append(track)
    midi.write('{}.mid'.format(name))


class TrainingVAE(TrainingInterface):

    def _batch_to_inputs(self, batch):
        #
        # self.save_midi(batch['pr_mats'][0], 'p0')
        # self.save_midi(batch['pr_mats_voicing'][0], 'v0')
        # self.save_midi(batch['pr_mats'][1], 'p1')
        # self.save_midi(batch['pr_mats_voicing'][1], 'v1')
        # self.save_midi(batch['pr_mats'][2], 'p2')
        # self.save_midi(batch['pr_mats_voicing'][2], 'v2')
        # input()

        if 'pr_mats_voicing' not in batch:
            return batch['p_grids'].to(self.device).long(), \
                   batch['chord'].to(self.device).float(), \
                   batch['pr_mats'].to(self.device).float(), \
                   torch.tensor(batch['dt_x']).to(self.device).float()
        else:
            if 'voicing_multi_hot' not in batch:
                return batch['p_grids_voicing'].to(self.device).long(), \
                       batch['chord'].to(self.device).float(), \
                       batch['pr_mats_voicing'].to(self.device).float(), \
                       batch['p_grids'].to(self.device).long(), \
                       batch['pr_mats'].to(self.device).float(), \
                       torch.tensor(batch['dt_x']).to(self.device).float()
            else:
                return batch['p_grids'].to(self.device).long(), \
                       batch['p_grids_voicing'].to(self.device).long(), \
                       batch['pr_mats'].to(self.device).float(), \
                       batch['pr_mats_voicing'].to(self.device).float(), \
                       batch['voicing_multi_hot'].to(self.device).float(), \
                       torch.tensor(batch['dt_x']).to(self.device).float()
