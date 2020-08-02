from dataset import prepare_dataset
from amc_dl.torch_plus import DataLoaders
from amc_dl.torch_plus import TrainingInterface


class MusicDataLoaders(DataLoaders):

    @staticmethod
    def get_loaders(seed, bs_train, bs_val,
                    portion=8, shift_low=-6, shift_high=5, num_bar=2,
                    contain_chord=True, random_train=True, random_val=False):
        train, val = prepare_dataset(seed, bs_train, bs_val, portion, shift_low,
                                     shift_high, num_bar, contain_chord,
                                     random_train, random_val)
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
        _, _, pr_mat, x, c, dt_x = batch
        pr_mat = pr_mat.to(self.device).float()
        x = x.to(self.device).long()
        c = c.to(self.device).float()
        dt_x = dt_x.to(self.device).float()
        return x, c, pr_mat, dt_x

