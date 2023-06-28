from data_utils.dataset import prepare_dataset_niko, prepare_dataset, prepare_dataset_pop909_voicing, \
    prepare_dataset_pop909_stage_a
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
                                                        shift_high, num_bar, random_train, random_val)
        elif dataset_name == 'pop909':
            train, val = prepare_dataset(seed, bs_train, bs_val, portion, shift_low,
                                         shift_high, num_bar, random_train, random_val)
        elif dataset_name == 'pop909_stage_a':
            train, val = prepare_dataset_pop909_stage_a(seed, bs_train, bs_val, portion, shift_low,
                                              shift_high, num_bar, random_train, random_val, full_song)
        else:
            raise Exception
        return MusicDataLoaders(train, val, bs_train, bs_val)


class TrainingVAE(TrainingInterface):

    def _batch_to_inputs(self, batch):

        if 'pr_mats_voicing' not in batch:
            return batch['p_grids'].to(self.device).long(), \
                   batch['chord'].to(self.device).float(), \
                   batch['pr_mats'].to(self.device).float(), \
                   batch['dt_x'].to(self.device).float()
        else:
            return batch['p_grids'].to(self.device).long(), \
                   batch['p_grids_voicing'].to(self.device).long(), \
                   batch['pr_mats'].to(self.device).float(), \
                   batch['pr_mats_voicing'].to(self.device).float(), \
                   batch['voicing_multi_hot'].to(self.device).float(), \
                   batch['dt_x'].to(self.device).float()
