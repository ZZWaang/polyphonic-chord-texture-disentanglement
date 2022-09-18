from dataset import prepare_dataset_niko, prepare_dataset, prepare_dataset_pop909_voicing
from amc_dl.torch_plus import DataLoaders
from amc_dl.torch_plus import TrainingInterface


class MusicDataLoaders(DataLoaders):

    @staticmethod
    def get_loaders(seed, dataset_name, bs_train, bs_val,
                    portion=8, shift_low=-6, shift_high=5, num_bar=2,
                    contain_chord=True, random_train=True, random_val=False):
        if dataset_name == 'niko':
            train, val = prepare_dataset_niko(seed, bs_train, bs_val, portion, shift_low,
                                              shift_high, num_bar, random_train, random_val)
        elif dataset_name == 'pop909_voicing':
            train, val = prepare_dataset_pop909_voicing(seed, bs_train, bs_val, portion, shift_low,
                                                        shift_high, num_bar, random_train, random_val)
        elif dataset_name == 'pop909':
            train, val = prepare_dataset(seed, bs_train, bs_val, portion, shift_low,
                                         shift_high, num_bar, random_train, random_val)
        else:
            raise Exception
        return MusicDataLoaders(train, val, bs_train, bs_val)

    def batch_to_inputs(self, batch):
        _, _, pr_mat, x, c, dt_x = batch
        pr_mat = pr_mat.to(self.device).float()
        x = x.to(self.device).long()
        c = c.to(self.device).float()
        dt_x = dt_x.to(self.device).float()
        return x, c, pr_mat, dt_x


class TrainingVAE(TrainingInterface):

    def _batch_to_inputs(self, batch):

        assert len(batch) == 6 or len(batch) == 8
        if len(batch) == 6:
            _, _, pr_mat, x, c, dt_x = batch
            pr_mat = pr_mat.to(self.device).float()
            x = x.to(self.device).long()
            c = c.to(self.device).float()
            dt_x = dt_x.to(self.device).float()
            return x, c, pr_mat, dt_x
        else:
            _, _, pr_mat, x, c, dt_x, pr_mat_voicing, x_voicing = batch
            pr_mat_voicing = pr_mat_voicing.to(self.device).float()
            pr_mat = pr_mat.to(self.device).float()
            x = x.to(self.device).long()
            x_voicing = x_voicing.to(self.device).long()
            dt_x = dt_x.to(self.device).float()
            return x, x_voicing, pr_mat, pr_mat_voicing, dt_x
