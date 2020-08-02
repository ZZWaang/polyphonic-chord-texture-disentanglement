import warnings

warnings.simplefilter('ignore', UserWarning)
from model import DisentangleVAE
from ptvae import RnnEncoder, TextureEncoder, PtvaeEncoder, PtvaeDecoder, \
    RnnDecoder
from dataset_loaders import MusicDataLoaders, TrainingVAE
from dataset import SEED
from amc_dl.torch_plus import LogPathManager, SummaryWriters, \
    ParameterScheduler, OptimizerScheduler, MinExponentialLR, \
    TeacherForcingScheduler, ConstantScheduler
from amc_dl.torch_plus.train_utils import kl_anealing
import torch
from torch import optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
readme_fn = './train.py'
batch_size = 128
n_epoch = 6
clip = 1
parallel = False
weights = [1, 0.5]
beta = 0.1
tf_rates = [(0.6, 0), (0.5, 0), (0.5, 0)]
lr = 1e-3
name = 'disvae-nozoth'

parallel = parallel if torch.cuda.is_available() and \
                       torch.cuda.device_count() > 1 else False
# train_model
chd_encoder = RnnEncoder(36, 1024, 256)
rhy_encoder = PtvaeEncoder(device=device, z_size=256, max_pitch=39 - 8, min_pitch=0)
# rhy_encoder = TextureEncoder(256, 1024, 256)
# pt_encoder = PtvaeEncoder(device=device, z_size=152)
chd_decoder = RnnDecoder(z_dim=256)
pt_decoder = PtvaeDecoder(note_embedding=None,
                          dec_dur_hid_size=64, z_size=512)
model = DisentangleVAE(name, device, chd_encoder,
                       rhy_encoder, pt_decoder, chd_decoder)

# data loaders
data_loaders = \
    MusicDataLoaders.get_loaders(SEED, bs_train=batch_size, bs_val=batch_size,
                                 portion=8, shift_low=-6, shift_high=5,
                                 num_bar=2,
                                 contain_chord=True)

log_path_mng = LogPathManager(readme_fn)

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = MinExponentialLR(optimizer, gamma=0.9999, minimum=1e-5)
optimizer_scheduler = OptimizerScheduler(optimizer, scheduler, clip)

writer_names = ['loss', 'recon_loss', 'pl', 'dl', 'kl_loss', 'kl_chd',
                'kl_rhy', 'chord_loss', 'root_loss', 'chroma_loss', 'bass_loss']
#, 'chord', 'root', 'chroma', 'bass']
tags = {'loss': None}
summary_writers = SummaryWriters(writer_names, tags, log_path_mng.writer_path)
tfr1_scheduler = TeacherForcingScheduler(*tf_rates[0])
tfr2_scheduler = TeacherForcingScheduler(*tf_rates[1])
tfr3_scheduler = TeacherForcingScheduler(*tf_rates[2])
weights_scheduler = ConstantScheduler(weights)
beta_scheduler = TeacherForcingScheduler(beta, 0., f=kl_anealing)
params_dic = dict(tfr1=tfr1_scheduler, tfr2=tfr2_scheduler,
                  tfr3=tfr3_scheduler,
                  beta=beta_scheduler, weights=weights_scheduler)
param_scheduler = ParameterScheduler(**params_dic)

training = TrainingVAE(device, model, parallel, log_path_mng,
                       data_loaders, summary_writers, optimizer_scheduler,
                       param_scheduler, n_epoch)
training.run()

if __name__ == '__main__':
    pass
