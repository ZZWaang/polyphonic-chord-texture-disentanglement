import argparse
import warnings

from latentAR import zTransformer, InfoNCELoss
from models.model import DisentangleVAE, DisentangleVoicingTextureVAE, DisentangleARG, DisentangleARGStageB, \
    DisentangleARGFull
from models.ptvae import RnnEncoder, TextureEncoder, PtvaeDecoder, RnnDecoder, NoteSummaryAttention, \
    PtvaeAttentionDecoder
from data_utils.dataset_loaders import MusicDataLoaders, TrainingVAE
from data_utils.dataset import SEED
from amc_dl.torch_plus import LogPathManager, SummaryWriters, ParameterScheduler, OptimizerScheduler, \
    MinExponentialLR, TeacherForcingScheduler, ConstantScheduler
from amc_dl.torch_plus.train_utils import kl_anealing
import torch
from torch import optim
import config

warnings.simplefilter('ignore', UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default=config.device, choices=['cuda', 'cpu'])
parser.add_argument('--readme_fn', type=str, default=config.readme_fn)
parser.add_argument('--batch_size', type=int, default=config.batch_size)
parser.add_argument('--epoch', type=int, default=config.n_epoch)
parser.add_argument('--clip', type=int, default=config.clip)
parser.add_argument('--parallel', type=bool, default=config.parallel)
parser.add_argument('--training_stage', type=int, default=config.training_stage)
parser.add_argument('--name', type=str, default=config.name)
parser.add_argument('--attention_emb', type=int, default=config.attention_emb)

current_config = parser.parse_args()
config.device = torch.device(current_config.device) if torch.cuda.is_available() else torch.device('cpu')
config.readme_fn = current_config.readme_fn
config.batch_size = current_config.batch_size
config.epoch = current_config.epoch
config.clip = current_config.clip
config.training_stage = current_config.training_stage
config.name = current_config.name
config.parallel = current_config.parallel if torch.cuda.is_available() and \
                                             torch.cuda.device_count() > 1 else False
config.attention_emb = current_config.attention_emb

# stage a polydis
if config.training_stage == 1:
    chd_encoder = RnnEncoder(36, 1024, 256)
    voicing_encoder = TextureEncoder(256, 1024, 256)
    chd_decoder = RnnDecoder(z_dim=256)
    voicing_decoder = PtvaeDecoder(note_embedding=None,
                                   dec_dur_hid_size=64, z_size=512)
    model = DisentangleVAE(config.name, config.device, chd_encoder,
                           voicing_encoder, voicing_decoder, chd_decoder)
    data_loaders = MusicDataLoaders.get_loaders(SEED, dataset_name='pop909_stage_a',
                                                bs_train=config.batch_size, bs_val=config.batch_size,
                                                portion=8, shift_low=-6, shift_high=5,
                                                num_bar=2, contain_chord=True)
    writer_names = ['loss', 'recon_loss', 'pl', 'dl', 'kl_loss', 'kl_chd',
                    'kl_rhy', 'chord_loss', 'root_loss', 'chroma_loss', 'bass_loss']

# stage b polydis
elif config.training_stage == 2:
    voicing_encoder = TextureEncoder(256, 1024, 256)
    rhy_encoder = TextureEncoder(256, 1024, 256)
    voicing_decoder = PtvaeDecoder(note_embedding=None,
                                   dec_dur_hid_size=64, z_size=256)
    pt_decoder = PtvaeAttentionDecoder(note_embedding=None,
                                       dec_dur_hid_size=64, z_size=512, attention_emb=32)
    model = DisentangleVoicingTextureVAE(config.name, config.device, voicing_encoder,
                                         rhy_encoder, pt_decoder, voicing_decoder)
    data_loaders = MusicDataLoaders.get_loaders(SEED, dataset_name='pop909_voicing',
                                                bs_train=config.batch_size, bs_val=config.batch_size,
                                                portion=8, shift_low=-6, shift_high=5,
                                                num_bar=8, contain_chord=True)
    writer_names = ['loss', 'recon_loss', 'pl', 'dl', 'kl_loss', 'kl_chd',
                    'kl_rhy', 'recon_loss_c', 'pl_c', 'dl_c']

# stage a arg
elif config.training_stage == 3:
    chd_encoder = RnnEncoder(36, 1024, 256)
    for p in chd_encoder.parameters():
        p.requires_grad = False
    voicing_encoder = TextureEncoder(256, 1024, 256)
    for p in voicing_encoder.parameters():
        p.requires_grad = False
    chd_decoder = RnnDecoder(z_dim=256)
    for p in chd_decoder.parameters():
        p.requires_grad = False
    voicing_decoder = PtvaeDecoder(note_embedding=None,
                                   dec_dur_hid_size=64, z_size=512)
    for p in voicing_decoder.parameters():
        p.requires_grad = False
    arg_decoder = zTransformer(dim_model=512, num_heads=8, num_decoder_layers=12, dropout_p=0.1)
    arg_loss = InfoNCELoss(input_dim=512, sample_dim=512, skip_projection=False)
    model = DisentangleARG(config.name, config.device, chd_encoder,
                           voicing_encoder, voicing_decoder, chd_decoder, arg_decoder, arg_loss)
    model.load_state_dict(
        torch.load('result_2023-06-06_122449/models/disvae-nozoth_final.pt', map_location=config.device), strict=False)
    data_loaders = MusicDataLoaders.get_loaders(SEED, dataset_name='pop909_stage_a',
                                                bs_train=config.batch_size, bs_val=config.batch_size,
                                                portion=8, shift_low=-6, shift_high=5,
                                                num_bar=2, contain_chord=True, full_song=True)
    writer_names = ['loss', 'recon_loss', 'pl', 'dl', 'chord_loss', 'root_loss', 'chroma_loss', 'bass_loss', 'arg_loss']

# stage b arg
elif config.training_stage == 4:
    voicing_encoder = TextureEncoder(256, 1024, 256)
    rhy_encoder = TextureEncoder(256, 1024, 256)
    voicing_decoder = PtvaeDecoder(note_embedding=None,
                                   dec_dur_hid_size=64, z_size=256)
    pt_decoder = PtvaeDecoder(note_embedding=None,
                              dec_dur_hid_size=64, z_size=512)
    for p in voicing_encoder.parameters():
        p.requires_grad = False
    for p in rhy_encoder.parameters():
        p.requires_grad = False
    for p in voicing_decoder.parameters():
        p.requires_grad = False
    for p in pt_decoder.parameters():
        p.requires_grad = False
    arg_decoder = zTransformer(dim_model=512, num_heads=8, num_decoder_layers=12, dropout_p=0.1)
    arg_loss = InfoNCELoss(input_dim=512, sample_dim=512, skip_projection=False)
    model = DisentangleARGStageB(config.name, config.device, voicing_encoder,
                                 rhy_encoder, pt_decoder, voicing_decoder, arg_decoder, arg_loss)
    model.load_state_dict(
        torch.load('data/train_stage2_20221009.pt', map_location=config.device), strict=False)
    data_loaders = MusicDataLoaders.get_loaders(SEED, dataset_name='pop909_voicing',
                                                bs_train=config.batch_size, bs_val=config.batch_size,
                                                portion=8, shift_low=-6, shift_high=5,
                                                num_bar=8, contain_chord=True, full_song=True)
    writer_names = ['loss', 'recon_loss', 'pl', 'dl', 'recon_loss_c', 'pl_c', 'dl_c', 'arg_loss']

# stage a+b arg
elif config.training_stage == 5:

    # stage a
    stage_a_chd_encoder = RnnEncoder(36, 1024, 256)
    for p in stage_a_chd_encoder.parameters():
        p.requires_grad = False
    stage_a_voicing_encoder = TextureEncoder(256, 1024, 256)
    for p in stage_a_voicing_encoder.parameters():
        p.requires_grad = False
    stage_a_chd_decoder = RnnDecoder(z_dim=256)
    for p in stage_a_chd_decoder.parameters():
        p.requires_grad = False
    stage_a_voicing_decoder = PtvaeDecoder(note_embedding=None,
                                           dec_dur_hid_size=64, z_size=512)
    for p in stage_a_voicing_decoder.parameters():
        p.requires_grad = False
    stage_a_arg_decoder = zTransformer(dim_model=512, num_heads=8, num_decoder_layers=6, dropout_p=0.1)
    stage_a_arg_loss = InfoNCELoss(input_dim=512, sample_dim=512, skip_projection=False)

    # stage b
    stage_b_voicing_encoder = TextureEncoder(256, 1024, 256)
    stage_b_rhy_encoder = TextureEncoder(256, 1024, 256)
    stage_b_voicing_decoder = PtvaeDecoder(note_embedding=None,
                                           dec_dur_hid_size=64, z_size=256)
    stage_b_pt_decoder = PtvaeDecoder(note_embedding=None,
                                      dec_dur_hid_size=64, z_size=512)
    for p in stage_b_voicing_encoder.parameters():
        p.requires_grad = False
    for p in stage_b_rhy_encoder.parameters():
        p.requires_grad = False
    for p in stage_b_voicing_decoder.parameters():
        p.requires_grad = False
    for p in stage_b_pt_decoder.parameters():
        p.requires_grad = False
    stage_b_arg_decoder = zTransformer(dim_model=512, num_heads=8, num_decoder_layers=6, dropout_p=0.1)
    stage_b_arg_loss = InfoNCELoss(input_dim=512, sample_dim=512, skip_projection=False)

    # final model
    model = DisentangleARGFull(config.name, config.device,
                               stage_a_chd_encoder, stage_a_voicing_encoder, stage_a_chd_decoder,
                               stage_a_voicing_decoder, stage_a_arg_decoder, stage_a_arg_loss,
                               stage_b_voicing_encoder, stage_b_rhy_encoder, stage_b_voicing_decoder,
                               stage_b_pt_decoder, stage_b_arg_decoder, stage_b_arg_loss)

    # stage dict
    stage_a_state_dict = torch.load('result_2023-06-06_122449/models/disvae-nozoth_final.pt',
                                    map_location=config.device)
    for key in list(stage_a_state_dict.keys()):
        stage_a_state_dict['stage_a_' + key] = stage_a_state_dict[key]

    stage_b_state_dict = torch.load('data/train_stage2_20221009.pt',
                                    map_location=config.device)
    for key in list(stage_b_state_dict.keys()):
        stage_b_state_dict['stage_b_' + key] = stage_b_state_dict[key]

    model.load_state_dict(stage_a_state_dict, strict=False)
    model.load_state_dict(stage_b_state_dict, strict=False)

    # data loader
    data_loaders = MusicDataLoaders.get_loaders(SEED, dataset_name='pop909_voicing',
                                                bs_train=config.batch_size, bs_val=config.batch_size,
                                                portion=8, shift_low=-6, shift_high=5,
                                                num_bar=8, contain_chord=True, full_song=True)

    # writer
    writer_names = ['loss', 'chord_loss', 'stage_a_recon_loss', 'stage_a_pl', 'stage_a_dl', 'stage_b_recon_loss',
                    'stage_b_pl', 'stage_b_dl', 'stage_a_arg_loss', 'stage_b_arg_loss']


else:
    raise Exception

log_path_mng = LogPathManager(config.readme_fn)

optimizer = optim.Adam(model.parameters(), lr=config.lr)
scheduler = MinExponentialLR(optimizer, gamma=0.9999, minimum=1e-5)
optimizer_scheduler = OptimizerScheduler(optimizer, scheduler, config.clip)

# , 'chord', 'root', 'chroma', 'bass']
tags = {'loss': None}
summary_writers = SummaryWriters(writer_names, tags, log_path_mng.writer_path)
tfr1_scheduler = TeacherForcingScheduler(*config.tf_rates[0])
tfr2_scheduler = TeacherForcingScheduler(*config.tf_rates[1])
tfr3_scheduler = TeacherForcingScheduler(*config.tf_rates[2])
weights_scheduler = ConstantScheduler(config.weights)
beta_scheduler = TeacherForcingScheduler(config.beta, 0., f=kl_anealing)
params_dic = dict(tfr1=tfr1_scheduler, tfr2=tfr2_scheduler,
                  tfr3=tfr3_scheduler,
                  beta=beta_scheduler, weights=weights_scheduler)
param_scheduler = ParameterScheduler(**params_dic)

training = TrainingVAE(config.device, model, config.parallel, log_path_mng,
                       data_loaders, summary_writers, optimizer_scheduler,
                       param_scheduler, config.n_epoch)
training.run()

if __name__ == '__main__':
    pass
