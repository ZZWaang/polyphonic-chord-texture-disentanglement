from amc_dl.torch_plus import PytorchModel
from amc_dl.torch_plus.train_utils import get_zs_from_dists, kl_with_normal
import torch
from torch import nn
from torch.distributions import Normal
import numpy as np

from latentAR import zTransformer, InfoNCELoss
from utils.utils import target_to_3dtarget, pr2midi
from models.ptvae import RnnEncoder, RnnDecoder, PtvaeDecoder, TextureEncoder, NoteSummaryAttention, \
    PtvaeAttentionDecoder


class DisentangleVAE(PytorchModel):

    def __init__(self, name, device, chd_encoder, rhy_encoder, decoder,
                 chd_decoder):
        super(DisentangleVAE, self).__init__(name, device)
        self.chd_encoder = chd_encoder
        self.rhy_encoder = rhy_encoder
        self.decoder = decoder
        self.num_step = self.decoder.num_step
        self.chd_decoder = chd_decoder

    def confuse_prmat(self, pr_mat):
        non_zero_ent = torch.nonzero(pr_mat.long())
        eps = torch.randint(0, 2, (non_zero_ent.size(0),))
        eps = ((2 * eps) - 1).long()
        confuse_ent = torch.clamp(non_zero_ent[:, 2] + eps, min=0, max=127)
        pr_mat[non_zero_ent[:, 0], non_zero_ent[:, 1], confuse_ent] = \
            pr_mat[non_zero_ent[:, 0], non_zero_ent[:, 1], non_zero_ent[:, 2]]
        return pr_mat

    def get_chroma(self, pr_mat):
        bs = pr_mat.size(0)
        pad = torch.zeros(bs, 32, 4).to(self.device)
        pr_mat = torch.cat([pr_mat, pad], dim=-1)
        c = pr_mat.view(bs, 32, -1, 12).contiguous()
        c = c.sum(dim=-2)  # (bs, 32, 12)
        c = c.view(bs, 8, 4, 12)
        c = c.sum(dim=-2).float()
        c = torch.log(c + 1)
        return c.to(self.device)

    def run(self, x, c, pr_mat, tfr1, tfr2, tfr3, confuse=True):
        embedded_x, lengths = self.decoder.emb_x(x)
        # cc = self.get_chroma(pr_mat)
        dist_chd = self.chd_encoder(c)
        # pr_mat = self.confuse_prmat(pr_mat)
        dist_rhy = self.rhy_encoder(pr_mat)
        z_chd, z_rhy = get_zs_from_dists([dist_chd, dist_rhy], True)
        dec_z = torch.cat([z_chd, z_rhy], dim=-1)
        pitch_outs, dur_outs = self.decoder(dec_z, False, embedded_x,
                                            lengths, tfr1, tfr2)
        recon_root, recon_chroma, recon_bass = self.chd_decoder(z_chd, False,
                                                                tfr3, c)
        return pitch_outs, dur_outs, dist_chd, dist_rhy, recon_root, \
               recon_chroma, recon_bass

    def loss_function(self, x, c, recon_pitch, recon_dur, dist_chd,
                      dist_rhy, recon_root, recon_chroma, recon_bass,
                      beta, weights, weighted_dur=False):
        recon_loss, pl, dl = self.decoder.recon_loss(x, recon_pitch, recon_dur,
                                                     weights, weighted_dur)
        kl_loss, kl_chd, kl_rhy = self.kl_loss(dist_chd, dist_rhy)
        chord_loss, root, chroma, bass = self.chord_loss(c, recon_root,
                                                         recon_chroma,
                                                         recon_bass)
        loss = recon_loss + beta * kl_loss + chord_loss
        return loss, recon_loss, pl, dl, kl_loss, kl_chd, kl_rhy, chord_loss, \
               root, chroma, bass

    def chord_loss(self, c, recon_root, recon_chroma, recon_bass):
        loss_fun = nn.CrossEntropyLoss()
        root = c[:, :, 0: 12].max(-1)[-1].view(-1).contiguous()
        chroma = c[:, :, 12: 24].long().view(-1).contiguous()
        bass = c[:, :, 24:].max(-1)[-1].view(-1).contiguous()

        recon_root = recon_root.view(-1, 12).contiguous()
        recon_chroma = recon_chroma.view(-1, 2).contiguous()
        recon_bass = recon_bass.view(-1, 12).contiguous()
        root_loss = loss_fun(recon_root, root)
        chroma_loss = loss_fun(recon_chroma, chroma)
        bass_loss = loss_fun(recon_bass, bass)
        chord_loss = root_loss + chroma_loss + bass_loss
        return chord_loss, root_loss, chroma_loss, bass_loss

    def kl_loss(self, *dists):
        # kl = kl_with_normal(dists[0])
        kl_chd = kl_with_normal(dists[0])
        kl_rhy = kl_with_normal(dists[1])
        kl_loss = kl_chd + kl_rhy
        return kl_loss, kl_chd, kl_rhy

    def loss(self, x, c, pr_mat, tfr1=0., tfr2=0., tfr3=0.,
             beta=0.1, weights=(1, 0.5)):
        outputs = self.run(x, c, pr_mat, tfr1, tfr2, tfr3)
        loss = self.loss_function(x, c, *outputs, beta, weights)
        return loss

    def inference_encode(self, pr_mat, c):
        self.eval()
        with torch.no_grad():
            dist_chd = self.chd_encoder(c)
            dist_rhy = self.rhy_encoder(pr_mat)
        return dist_chd, dist_rhy

    def inference_decode(self, z_chd, z_rhy):
        self.eval()
        with torch.no_grad():
            dec_z = torch.cat([z_chd, z_rhy], dim=-1)
            pitch_outs, dur_outs = self.decoder(dec_z, True, None,
                                                None, 0., 0.)
            est_x, _, _ = self.decoder.output_to_numpy(pitch_outs, dur_outs)
        return est_x

    def inference(self, pr_mat, c, sample):
        self.eval()
        with torch.no_grad():
            dist_chd = self.chd_encoder(c)
            dist_rhy = self.rhy_encoder(pr_mat)
            z_chd, z_rhy = get_zs_from_dists([dist_chd, dist_rhy], sample)
            dec_z = torch.cat([z_chd, z_rhy], dim=-1)
            pitch_outs, dur_outs = self.decoder(dec_z, True, None,
                                                None, 0., 0.)
            est_x, _, _ = self.decoder.output_to_numpy(pitch_outs, dur_outs)
        return est_x

    def inference_with_loss(self, pr_mat, c, sample):
        self.eval()
        with torch.no_grad():
            dist_chd = self.chd_encoder(c)
            dist_rhy = self.rhy_encoder(pr_mat)
            z_chd, z_rhy = get_zs_from_dists([dist_chd, dist_rhy], sample)
            dec_z = torch.cat([z_chd, z_rhy], dim=-1)
            pitch_outs, dur_outs = self.decoder(dec_z, True, None,
                                                None, 0., 0.)
            recon_root, recon_chroma, recon_bass = self.chd_decoder(z_chd, False,
                                                                    0., c)
            est_x, _, _ = self.decoder.output_to_numpy(pitch_outs, dur_outs)

            x = np.array([target_to_3dtarget(i,
                                             max_note_count=16,
                                             max_pitch=128,
                                             min_pitch=0,
                                             pitch_pad_ind=130,
                                             pitch_sos_ind=128,
                                             pitch_eos_ind=129) for i in pr_mat.cpu()])
            if torch.cuda.is_available():
                x = torch.from_numpy(x).type(torch.LongTensor).cuda()
            else:
                x = torch.from_numpy(x)

        def loss_for_inference(x, c, beta=0.1, weights=(1, 0.5)):
            outputs = pitch_outs, dur_outs, dist_chd, dist_rhy, recon_root, \
                      recon_chroma, recon_bass
            loss = self.loss_function(x, c, *outputs, beta, weights)
            return loss

        return est_x, loss_for_inference(x, c)

    def inference_save_z(self, pr_mat, c, sample, z_path):
        self.eval()
        with torch.no_grad():
            dist_chd = self.chd_encoder(c)
            dist_rhy = self.rhy_encoder(pr_mat)
            z_chd, z_rhy = get_zs_from_dists([dist_chd, dist_rhy], sample)
            dec_z = torch.cat([z_chd, z_rhy], dim=-1)
            torch.save(dec_z, z_path)

    def inference_only_decode(self, z, with_chord=False):
        self.eval()
        with torch.no_grad():
            pitch_outs, dur_outs = self.decoder(z, True, None,
                                                None, 0., 0.)
            est_x, _, _ = self.decoder.output_to_numpy(pitch_outs, dur_outs)
            if with_chord:
                z_chord = z[:, :256]
                print(z_chord.shape)
                recon_root, recon_chroma, recon_bass = self.chd_decoder(z_chord, True,
                                                                        0., None)
                return est_x, recon_root, recon_chroma, recon_bass
            return est_x

    def swap(self, pr_mat1, pr_mat2, c1, c2, fix_rhy, fix_chd):
        pr_mat = pr_mat1 if fix_rhy else pr_mat2
        c = c1 if fix_chd else c2
        est_x = self.inference(pr_mat, c, sample=False)
        return est_x

    def posterior_sample(self, pr_mat, c, scale=None, sample_chd=True,
                         sample_txt=True):
        if scale is None and sample_chd and sample_txt:
            est_x = self.inference(pr_mat, c, sample=True)
        else:
            dist_chd, dist_rhy = self.inference_encode(pr_mat, c)
            if scale is not None:
                mean_chd = dist_chd.mean
                mean_rhy = dist_rhy.mean
                # std_chd = torch.ones_like(dist_chd.mean) * scale
                # std_rhy = torch.ones_like(dist_rhy.mean) * scale
                std_chd = dist_chd.scale * scale
                std_rhy = dist_rhy.scale * scale
                dist_rhy = Normal(mean_rhy, std_rhy)
                dist_chd = Normal(mean_chd, std_chd)
            z_chd, z_rhy = get_zs_from_dists([dist_chd, dist_rhy], True)
            if not sample_chd:
                z_chd = dist_chd.mean
            if not sample_txt:
                z_rhy = dist_rhy.mean
            est_x = self.inference_decode(z_chd, z_rhy)
        return est_x

    def prior_sample(self, x, c, sample_chd=False, sample_rhy=False,
                     scale=1.):
        dist_chd, dist_rhy = self.inference_encode(x, c)
        mean = torch.zeros_like(dist_rhy.mean)
        loc = torch.ones_like(dist_rhy.mean) * scale
        if sample_chd:
            dist_chd = Normal(mean, loc)
        if sample_rhy:
            dist_rhy = Normal(mean, loc)
        z_chd, z_rhy = get_zs_from_dists([dist_chd, dist_rhy], True)
        return self.inference_decode(z_chd, z_rhy)

    def gt_sample(self, x):
        out = x[:, :, 1:].numpy()
        return out

    def interp(self, pr_mat1, c1, pr_mat2, c2, interp_chd=False,
               interp_rhy=False, int_count=10):
        dist_chd1, dist_rhy1 = self.inference_encode(pr_mat1, c1)
        dist_chd2, dist_rhy2 = self.inference_encode(pr_mat2, c2)
        [z_chd1, z_rhy1, z_chd2, z_rhy2] = \
            get_zs_from_dists([dist_chd1, dist_rhy1, dist_chd2, dist_rhy2],
                              False)
        if interp_chd:
            z_chds = self.interp_z(z_chd1, z_chd2, int_count)
        else:
            z_chds = z_chd1.unsqueeze(1).repeat(1, int_count, 1)
        if interp_rhy:
            z_rhys = self.interp_z(z_rhy1, z_rhy2, int_count)
        else:
            z_rhys = z_rhy1.unsqueeze(1).repeat(1, int_count, 1)
        bs = z_chds.size(0)
        z_chds = z_chds.view(bs * int_count, -1).contiguous()
        z_rhys = z_rhys.view(bs * int_count, -1).contiguous()
        estxs = self.inference_decode(z_chds, z_rhys)
        return estxs.reshape((bs, int_count, 32, 15, -1))

    def interp_z(self, z1, z2, int_count=10):
        z1 = z1.numpy()
        z2 = z2.numpy()
        zs = torch.stack([self.interp_path(zz1, zz2, int_count)
                          for zz1, zz2 in zip(z1, z2)], dim=0)
        return zs

    def interp_path(self, z1, z2, interpolation_count=10):
        result_shape = z1.shape
        z1 = z1.reshape(-1)
        z2 = z2.reshape(-1)

        def slerp2(p0, p1, t):
            omega = np.arccos(
                np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)))
            so = np.sin(omega)
            return np.sin((1.0 - t) * omega)[:, None] / so * p0[
                None] + np.sin(
                t * omega)[:, None] / so * p1[None]

        percentages = np.linspace(0.0, 1.0, interpolation_count)

        normalized_z1 = z1 / np.linalg.norm(z1)
        normalized_z2 = z2 / np.linalg.norm(z2)
        dirs = slerp2(normalized_z1, normalized_z2, percentages)
        length = np.linspace(np.log(np.linalg.norm(z1)),
                             np.log(np.linalg.norm(z2)),
                             interpolation_count)
        out = (dirs * np.exp(length[:, None])).reshape(
            [interpolation_count] + list(result_shape))
        # out = np.array([(1 - t) * z1 + t * z2 for t in percentages])
        return torch.from_numpy(out).to(self.device).float()

    @staticmethod
    def init_model(device=None, chd_size=256, txt_size=256, num_channel=10):
        name = 'disvae'
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available()
                                  else 'cpu')
        # chd_encoder = RnnEncoder(36, 1024, 256)
        chd_encoder = RnnEncoder(36, 1024, chd_size)
        # rhy_encoder = TextureEncoder(256, 1024, 256)
        rhy_encoder = TextureEncoder(256, 1024, txt_size, num_channel)
        # pt_encoder = PtvaeEncoder(device=device, z_size=152)
        # chd_decoder = RnnDecoder(z_dim=256)
        chd_decoder = RnnDecoder(z_dim=chd_size)
        # pt_decoder = PtvaeDecoder(note_embedding=None,
        #                           dec_dur_hid_size=64, z_size=512)
        pt_decoder = PtvaeDecoder(note_embedding=None,
                                  dec_dur_hid_size=64,
                                  z_size=chd_size + txt_size)

        model = DisentangleVAE(name, device, chd_encoder,
                               rhy_encoder, pt_decoder, chd_decoder)
        return model


class DisentangleVoicingTextureVAE(PytorchModel):

    def __init__(self, name, device, voicing_encoder, rhy_encoder, decoder,
                 voicing_decoder):
        super(DisentangleVoicingTextureVAE, self).__init__(name, device)
        self.voicing_encoder = voicing_encoder
        self.rhy_encoder = rhy_encoder
        self.decoder = decoder
        self.voicing_decoder = voicing_decoder

    def loss(self, x, c, pr_mat, pr_mat_c, voicing_multi_hot, tfr1=0., tfr2=0., tfr3=0.,
             beta=0.1, weights=(1, 0.5)):
        outputs = self.run(x, c, pr_mat, pr_mat_c, voicing_multi_hot, tfr1, tfr2, tfr3)
        loss = self.loss_function(x, c, *outputs, beta, weights)
        return loss

    def inference(self, pr_mat, c, sample, save_z=None):
        self.eval()
        with torch.no_grad():
            dist_chd = self.voicing_encoder(c)
            dist_rhy = self.rhy_encoder(pr_mat)
            z_chd, z_rhy = get_zs_from_dists([dist_chd, dist_rhy], sample)
            dec_z = torch.cat([z_chd, z_rhy], dim=-1)
            if save_z:
                torch.save(dec_z, 'zs/s2/{}.pt'.format(save_z))
            pitch_outs, dur_outs = self.decoder(dec_z, True, None,
                                                None, 0., 0.)
            est_x, _, _ = self.decoder.output_to_numpy(pitch_outs, dur_outs)
        return est_x

    def inference_with_chord_decode(self, pr_mat, c, vm, sample, save_z=None):
        self.eval()
        with torch.no_grad():
            dist_chd = self.voicing_encoder(c)
            dist_rhy = self.rhy_encoder(pr_mat)
            z_chd, z_rhy = get_zs_from_dists([dist_chd, dist_rhy], sample)
            dec_z = torch.cat([z_chd, z_rhy], dim=-1)
            if save_z:
                torch.save(dec_z, 'zs/s2/{}.pt'.format(save_z))
            pitch_outs, dur_outs = self.decoder(dec_z, True, None,
                                                None, 0., 0.)
            pitch_outs_c, dur_outs_c = self.voicing_decoder(z_chd, True, None,
                                                            None, 0., 0.)
            est_x, _, _ = self.decoder.output_to_numpy(pitch_outs, dur_outs)
            est_x_c, _, _ = self.voicing_decoder.output_to_numpy(pitch_outs_c, dur_outs_c)
        return est_x, est_x_c

    def loss_function(self, x, c, recon_pitch, recon_dur, dist_chd,
                      dist_rhy, recon_pitch_c, recon_dur_c,
                      beta, weights, weighted_dur=False):
        recon_loss, pl, dl = self.decoder.recon_loss(x, recon_pitch, recon_dur,
                                                     weights, weighted_dur)
        kl_loss, kl_chd, kl_rhy = self.kl_loss(dist_chd, dist_rhy)
        recon_loss_c, pl_c, dl_c = self.voicing_decoder.recon_loss(c, recon_pitch_c, recon_dur_c, weights, weighted_dur)
        loss = recon_loss + beta * kl_loss + recon_loss_c
        return loss, recon_loss, pl, dl, kl_loss, kl_chd, kl_rhy, recon_loss_c, pl_c, dl_c

    def kl_loss(self, *dists):
        # kl = kl_with_normal(dists[0])
        kl_chd = kl_with_normal(dists[0])
        kl_rhy = kl_with_normal(dists[1])
        kl_loss = kl_chd + kl_rhy
        return kl_loss, kl_chd, kl_rhy

    @staticmethod
    def init_model(device=None, voicing_size=256, txt_size=256, num_channel=10):
        name = 'disvae2'
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available()
                                  else 'cpu')
        # chd_encoder = RnnEncoder(36, 1024, 256)
        voicing_encoder = TextureEncoder(256, 1024, txt_size, num_channel)
        # rhy_encoder = TextureEncoder(256, 1024, 256)
        rhy_encoder = TextureEncoder(256, 1024, txt_size, num_channel)
        # pt_encoder = PtvaeEncoder(device=device, z_size=152)
        # chd_decoder = RnnDecoder(z_dim=256)
        voicing_decoder = PtvaeDecoder(note_embedding=None,
                                       dec_dur_hid_size=64,
                                       z_size=voicing_size)
        pt_decoder = PtvaeDecoder(note_embedding=None,
                                  dec_dur_hid_size=64, z_size=512)
        # pt_decoder = PtvaeAttentionDecoder(note_embedding=None,
        #                                    dec_dur_hid_size=64,
        #                                    z_size=voicing_size + txt_size,
        #                                    attention_emb=32)

        model = DisentangleVoicingTextureVAE(name, device, voicing_encoder,
                                             rhy_encoder, pt_decoder, voicing_decoder)
        return model

    def run(self, x, c, pr_mat, pr_mat_c, voicing_multi_hot, tfr1, tfr2, tfr3, confuse=True):
        embedded_x, lengths = self.decoder.emb_x(x)
        embedded_c, lengths_c = self.voicing_decoder.emb_x(c)
        # cc = self.get_chroma(pr_mat)
        dist_voicing = self.voicing_encoder(pr_mat_c)
        # pr_mat = self.confuse_prmat(pr_mat)
        dist_rhy = self.rhy_encoder(pr_mat)
        z_voicing, z_rhy = get_zs_from_dists([dist_voicing, dist_rhy], True)
        dec_z = torch.cat([z_voicing, z_rhy], dim=-1)
        pitch_outs, dur_outs = self.decoder(dec_z, False, embedded_x, voicing_multi_hot, lengths, tfr1, tfr2)
        pitch_outs_c, dur_outs_c = self.voicing_decoder(z_voicing, False, embedded_c, lengths_c, tfr1, tfr2)
        return pitch_outs, dur_outs, dist_voicing, dist_rhy, pitch_outs_c, dur_outs_c


class DisentangleARG(PytorchModel):

    def __init__(self, name, device, chd_encoder, rhy_encoder, decoder,
                 chd_decoder, arg_decoder, arg_loss_fun):
        super(DisentangleARG, self).__init__(name, device)
        self.chd_encoder = chd_encoder
        self.rhy_encoder = rhy_encoder
        self.decoder = decoder
        self.num_step = self.decoder.num_step
        self.chd_decoder = chd_decoder
        self.arg_decoder = arg_decoder
        self.arg_loss_fun = arg_loss_fun

    def run(self, x, c, pr_mat, tfr1, tfr2, tfr3, confuse=True):

        dist_chd = self.chd_encoder(c)
        dist_rhy = self.rhy_encoder(pr_mat)
        z_chd, z_rhy = get_zs_from_dists([dist_chd, dist_rhy], False)
        dec_z = torch.cat([z_chd, z_rhy], dim=-1).unsqueeze(0)

        y_input = dec_z[:, :-1]
        y_expected = dec_z[:, 1:]
        pred = self.arg_decoder(y_input)
        pred = pred[0]
        positive = torch.permute(y_expected, (1, 0, 2))
        negative = torch.concat([torch.unsqueeze(torch.cat((positive[: i, 0], positive[i + 1:, 0]), dim=0), dim=0) \
                                 for i in range(positive.shape[0])], dim=0)

        embedded_x, lengths = self.decoder.emb_x(x[1:])

        pitch_outs, dur_outs = self.decoder(pred, False, embedded_x, lengths, tfr1, tfr2)
        recon_root, recon_chroma, recon_bass = self.chd_decoder(pred[:, 0:256], False, tfr3, c[1:])
        return pitch_outs, dur_outs, recon_root, recon_chroma, recon_bass, pred, positive, negative

    def loss_function(self, x, c, recon_pitch, recon_dur, recon_root, recon_chroma, recon_bass, pred, positive,
                      negative,
                      beta, weights, weighted_dur=False):
        recon_loss, pl, dl = self.decoder.recon_loss(x[1:], recon_pitch, recon_dur,
                                                     weights, weighted_dur)
        chord_loss, root, chroma, bass = self.chord_loss(c[1:], recon_root,
                                                         recon_chroma,
                                                         recon_bass)
        arg_loss = self.arg_loss(pred, positive, negative, temperature=1)
        loss = recon_loss + chord_loss + arg_loss
        return loss, recon_loss, pl, dl, chord_loss, root, chroma, bass, arg_loss

    def chord_loss(self, c, recon_root, recon_chroma, recon_bass):
        loss_fun = nn.CrossEntropyLoss()
        root = c[:, :, 0: 12].max(-1)[-1].view(-1).contiguous()
        chroma = c[:, :, 12: 24].long().view(-1).contiguous()
        bass = c[:, :, 24:].max(-1)[-1].view(-1).contiguous()

        recon_root = recon_root.view(-1, 12).contiguous()
        recon_chroma = recon_chroma.view(-1, 2).contiguous()
        recon_bass = recon_bass.view(-1, 12).contiguous()
        root_loss = loss_fun(recon_root, root)
        chroma_loss = loss_fun(recon_chroma, chroma)
        bass_loss = loss_fun(recon_bass, bass)
        chord_loss = root_loss + chroma_loss + bass_loss
        return chord_loss, root_loss, chroma_loss, bass_loss

    def kl_loss(self, *dists):
        # kl = kl_with_normal(dists[0])
        kl_chd = kl_with_normal(dists[0])
        kl_rhy = kl_with_normal(dists[1])
        kl_loss = kl_chd + kl_rhy
        return kl_loss, kl_chd, kl_rhy

    def arg_loss(self, pred, positive, negative, temperature=1):
        return self.arg_loss_fun(pred, positive, negative, temperature)

    def loss(self, x, c, pr_mat, tfr1=0., tfr2=0., tfr3=0.,
             beta=0.1, weights=(1, 0.5)):
        x = x.squeeze(0)
        c = c.squeeze(0)
        pr_mat = pr_mat.squeeze(0)
        outputs = self.run(x, c, pr_mat, tfr1, tfr2, tfr3)
        loss = self.loss_function(x, c, *outputs, beta, weights)
        return loss

    def inference_encode(self, pr_mat, c):
        self.eval()
        with torch.no_grad():
            dist_chd = self.chd_encoder(c)
            dist_rhy = self.rhy_encoder(pr_mat)
        return dist_chd, dist_rhy

    def inference_decode(self, z_chd, z_rhy):
        self.eval()
        with torch.no_grad():
            dec_z = torch.cat([z_chd, z_rhy], dim=-1)
            pitch_outs, dur_outs = self.decoder(dec_z, True, None,
                                                None, 0., 0.)
            est_x, _, _ = self.decoder.output_to_numpy(pitch_outs, dur_outs)
        return est_x

    def inference(self, pr_mat, c, sample):
        self.eval()
        with torch.no_grad():
            dist_chd = self.chd_encoder(c)
            dist_rhy = self.rhy_encoder(pr_mat)
            z_chd, z_rhy = get_zs_from_dists([dist_chd, dist_rhy], sample)
            dec_z = torch.cat([z_chd, z_rhy], dim=-1).unsqueeze(0)
            pred = self.arg_decoder(dec_z)[0]
            all_est_x = []
            pitch_outs, dur_outs = self.decoder(pred, True, None,
                                                None, 0., 0.)
            est_x, _, _ = self.decoder.output_to_numpy(pitch_outs, dur_outs)
            all_est_x.append(est_x)
            for i in range(4):
                dec_z = torch.cat([dec_z.squeeze(0), pred[-1].unsqueeze(0)], dim=0)
                pred = self.arg_decoder(dec_z.unsqueeze(0))[0]
                pitch_outs, dur_outs = self.decoder(pred, True, None,
                                                    None, 0., 0.)
                est_x, _, _ = self.decoder.output_to_numpy(pitch_outs, dur_outs)
                all_est_x.append(est_x)

        return all_est_x

    def inference_with_loss(self, pr_mat, c, sample):
        self.eval()
        with torch.no_grad():
            dist_chd = self.chd_encoder(c)
            dist_rhy = self.rhy_encoder(pr_mat)
            z_chd, z_rhy = get_zs_from_dists([dist_chd, dist_rhy], sample)
            dec_z = torch.cat([z_chd, z_rhy], dim=-1)
            pitch_outs, dur_outs = self.decoder(dec_z, True, None,
                                                None, 0., 0.)
            recon_root, recon_chroma, recon_bass = self.chd_decoder(z_chd, False,
                                                                    0., c)
            est_x, _, _ = self.decoder.output_to_numpy(pitch_outs, dur_outs)

            x = np.array([target_to_3dtarget(i,
                                             max_note_count=16,
                                             max_pitch=128,
                                             min_pitch=0,
                                             pitch_pad_ind=130,
                                             pitch_sos_ind=128,
                                             pitch_eos_ind=129) for i in pr_mat.cpu()])
            if torch.cuda.is_available():
                x = torch.from_numpy(x).type(torch.LongTensor).cuda()
            else:
                x = torch.from_numpy(x)

        def loss_for_inference(x, c, beta=0.1, weights=(1, 0.5)):
            outputs = pitch_outs, dur_outs, dist_chd, dist_rhy, recon_root, \
                      recon_chroma, recon_bass
            loss = self.loss_function(x, c, *outputs, beta, weights)
            return loss

        return est_x, loss_for_inference(x, c)

    def inference_save_z(self, pr_mat, c, sample, z_path):
        self.eval()
        with torch.no_grad():
            dist_chd = self.chd_encoder(c)
            dist_rhy = self.rhy_encoder(pr_mat)
            z_chd, z_rhy = get_zs_from_dists([dist_chd, dist_rhy], sample)
            dec_z = torch.cat([z_chd, z_rhy], dim=-1)
            torch.save(dec_z, z_path)

    def inference_only_decode(self, z, with_chord=False):
        self.eval()
        with torch.no_grad():
            pitch_outs, dur_outs = self.decoder(z, True, None,
                                                None, 0., 0.)
            est_x, _, _ = self.decoder.output_to_numpy(pitch_outs, dur_outs)
            if with_chord:
                z_chord = z[:, :256]
                print(z_chord.shape)
                recon_root, recon_chroma, recon_bass = self.chd_decoder(z_chord, True,
                                                                        0., None)
                return est_x, recon_root, recon_chroma, recon_bass
            return est_x

    @staticmethod
    def init_model(device=None, chd_size=256, txt_size=256, num_channel=10):
        name = 'disvae'
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available()
                                  else 'cpu')
        # chd_encoder = RnnEncoder(36, 1024, 256)
        chd_encoder = RnnEncoder(36, 1024, chd_size)
        # rhy_encoder = TextureEncoder(256, 1024, 256)
        rhy_encoder = TextureEncoder(256, 1024, txt_size, num_channel)
        # pt_encoder = PtvaeEncoder(device=device, z_size=152)
        # chd_decoder = RnnDecoder(z_dim=256)
        chd_decoder = RnnDecoder(z_dim=chd_size)
        # pt_decoder = PtvaeDecoder(note_embedding=None,
        #                           dec_dur_hid_size=64, z_size=512)
        pt_decoder = PtvaeDecoder(note_embedding=None,
                                  dec_dur_hid_size=64,
                                  z_size=chd_size + txt_size)
        arg_decoder = zTransformer(dim_model=512, num_heads=8, num_decoder_layers=12, dropout_p=0.1)
        arg_loss_fun = InfoNCELoss(input_dim=512, sample_dim=512, skip_projection=False)
        model = DisentangleARG(name, device, chd_encoder,
                               rhy_encoder, pt_decoder, chd_decoder, arg_decoder, arg_loss_fun)
        return model


class DisentangleARGStageB(PytorchModel):
    def __init__(self, name, device, voicing_encoder, rhy_encoder, decoder,
                 voicing_decoder, arg_decoder, arg_loss_fun):
        super(DisentangleARGStageB, self).__init__(name, device)
        self.voicing_encoder = voicing_encoder
        self.rhy_encoder = rhy_encoder
        self.decoder = decoder
        self.voicing_decoder = voicing_decoder
        self.arg_decoder = arg_decoder
        self.arg_loss_fun = arg_loss_fun

    def loss(self, x, c, pr_mat, pr_mat_c, voicing_multi_hot, tfr1=0., tfr2=0., tfr3=0.,
             beta=0.1, weights=(1, 0.5)):
        x = x.squeeze(0)
        c = c.squeeze(0)
        pr_mat = pr_mat.squeeze(0)
        pr_mat_c = pr_mat_c.squeeze(0)
        outputs = self.run(x, c, pr_mat, pr_mat_c, voicing_multi_hot, tfr1, tfr2, tfr3)
        loss = self.loss_function(x, c, *outputs, beta, weights)
        return loss

    def inference(self, pr_mat, c, sample, save_z=None):
        self.eval()
        with torch.no_grad():
            dist_chd = self.voicing_encoder(c)
            dist_rhy = self.rhy_encoder(pr_mat)
            z_chd, z_rhy = get_zs_from_dists([dist_chd, dist_rhy], sample)
            dec_z = torch.cat([z_chd, z_rhy], dim=-1)
            if save_z:
                torch.save(dec_z, 'zs/s2/{}.pt'.format(save_z))
            pitch_outs, dur_outs = self.decoder(dec_z, True, None,
                                                None, 0., 0.)
            est_x, _, _ = self.decoder.output_to_numpy(pitch_outs, dur_outs)
        return est_x

    def inference_with_chord_decode(self, pr_mat, c, sample):
        self.eval()
        with torch.no_grad():
            dist_chd = self.voicing_encoder(c)
            dist_rhy = self.rhy_encoder(pr_mat)
            z_chd, z_rhy = get_zs_from_dists([dist_chd, dist_rhy], sample)
            dec_z = torch.cat([z_chd, z_rhy], dim=-1)

            pred = self.arg_decoder(dec_z)[0]
            all_est_x, all_est_c = [], []
            pitch_outs, dur_outs = self.decoder(pred, True, None,
                                                None, 0., 0.)
            pitch_outs_c, dur_outs_c = self.voicing_decoder(pred[:, 0:256], True, None,
                                                            None, 0., 0.)
            est_x, _, _ = self.decoder.output_to_numpy(pitch_outs, dur_outs)
            est_x_c, _, _ = self.voicing_decoder.output_to_numpy(pitch_outs_c, dur_outs_c)
            all_est_x.append(est_x)
            all_est_c.append(est_x_c)

            for i in range(4):
                dec_z = torch.cat([dec_z.squeeze(0), pred[-1].unsqueeze(0)], dim=0)
                pred = self.arg_decoder(dec_z.unsqueeze(0))[0]
                pitch_outs, dur_outs = self.decoder(pred[-1].unsqueeze(0), True, None,
                                                    None, 0., 0.)
                pitch_outs_c, dur_outs_c = self.voicing_decoder(pred[-1, 0:256].unsqueeze(0), True, None,
                                                                None, 0., 0.)
                est_x, _, _ = self.decoder.output_to_numpy(pitch_outs, dur_outs)
                est_x_c, _, _ = self.voicing_decoder.output_to_numpy(pitch_outs_c, dur_outs_c)
                all_est_x.append(est_x)
                all_est_c.append(est_x_c)

        return all_est_x, all_est_c

    def loss_function(self, x, c, recon_pitch, recon_dur, dist_chd,
                      dist_rhy, recon_pitch_c, recon_dur_c, pred, positive, negative,
                      beta, weights, weighted_dur=False):
        recon_loss, pl, dl = self.decoder.recon_loss(x[1:], recon_pitch, recon_dur,
                                                     weights, weighted_dur)
        recon_loss_c, pl_c, dl_c = self.voicing_decoder.recon_loss(c[1:], recon_pitch_c, recon_dur_c, weights,
                                                                   weighted_dur)
        arg_loss = self.arg_loss(pred, positive, negative, temperature=1)
        loss = recon_loss + recon_loss_c + arg_loss
        return loss, recon_loss, pl, dl, recon_loss_c, pl_c, dl_c, arg_loss

    def arg_loss(self, pred, positive, negative, temperature=1):
        return self.arg_loss_fun(pred, positive, negative, temperature)

    @staticmethod
    def init_model(device=None, voicing_size=256, txt_size=256, num_channel=10):
        name = 'disvae2'
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available()
                                  else 'cpu')
        voicing_encoder = TextureEncoder(256, 1024, 256)
        rhy_encoder = TextureEncoder(256, 1024, 256)
        voicing_decoder = PtvaeDecoder(note_embedding=None,
                                       dec_dur_hid_size=64, z_size=256)
        pt_decoder = PtvaeDecoder(note_embedding=None,
                                  dec_dur_hid_size=64, z_size=512)
        arg_decoder = zTransformer(dim_model=512, num_heads=8, num_decoder_layers=12, dropout_p=0.1)
        arg_loss = InfoNCELoss(input_dim=512, sample_dim=512, skip_projection=False)
        model = DisentangleARGStageB(name, device, voicing_encoder,
                                     rhy_encoder, pt_decoder, voicing_decoder, arg_decoder, arg_loss)
        return model

    def run(self, x, c, pr_mat, pr_mat_c, voicing_multi_hot, tfr1, tfr2, tfr3, confuse=True):

        embedded_x, lengths = self.decoder.emb_x(x[1:])
        embedded_c, lengths_c = self.voicing_decoder.emb_x(c[1:])
        dist_voicing = self.voicing_encoder(pr_mat_c)
        dist_rhy = self.rhy_encoder(pr_mat)
        z_voicing, z_rhy = get_zs_from_dists([dist_voicing, dist_rhy], False)
        dec_z = torch.cat([z_voicing, z_rhy], dim=-1).unsqueeze(0)
        y_input = dec_z[:, :-1]
        y_expected = dec_z[:, 1:]
        pred = self.arg_decoder(y_input)
        pred = pred[0]
        positive = torch.permute(y_expected, (1, 0, 2))
        negative = torch.concat([torch.unsqueeze(torch.cat((positive[: i, 0], positive[i + 1:, 0]), dim=0), dim=0) \
                                 for i in range(positive.shape[0])], dim=0)

        pitch_outs, dur_outs = self.decoder(pred, False, embedded_x, lengths, tfr1, tfr2)
        pitch_outs_c, dur_outs_c = self.voicing_decoder(pred[:, 0:256], False, embedded_c, lengths_c, tfr1, tfr2)
        return pitch_outs, dur_outs, dist_voicing, dist_rhy, pitch_outs_c, dur_outs_c, pred, positive, negative


class DisentangleARGFull(PytorchModel):
    def __init__(self, name, device, stage_a_chd_encoder, stage_a_voicing_encoder, stage_a_chd_decoder,
                 stage_a_voicing_decoder, stage_a_arg_decoder, stage_a_arg_loss,
                 stage_b_voicing_encoder, stage_b_rhy_encoder, stage_b_voicing_decoder,
                 stage_b_pt_decoder, stage_b_arg_decoder, stage_b_arg_loss):
        super(DisentangleARGFull, self).__init__(name, device)

        self.stage_a_chd_encoder = stage_a_chd_encoder
        self.stage_a_rhy_encoder = stage_a_voicing_encoder
        self.stage_a_decoder = stage_a_voicing_decoder
        self.stage_a_num_step = self.stage_a_decoder.num_step
        self.stage_a_chd_decoder = stage_a_chd_decoder
        self.stage_a_arg_decoder = stage_a_arg_decoder
        self.stage_a_arg_loss_fun = stage_a_arg_loss

        self.stage_b_voicing_encoder = stage_b_voicing_encoder
        self.stage_b_rhy_encoder = stage_b_rhy_encoder
        self.stage_b_decoder = stage_b_pt_decoder
        self.stage_b_voicing_decoder = stage_b_voicing_decoder
        self.stage_b_arg_decoder = stage_b_arg_decoder
        self.stage_b_arg_loss_fun = stage_b_arg_loss

    def loss(self, stage_a_x, stage_a_c, stage_a_pr_mat, stage_b_x, stage_b_pr_mat,
             tfr1=0., tfr2=0., tfr3=0., beta=0.1, weights=(1, 0.5)):

        stage_a_x, stage_a_c, stage_a_pr_mat, stage_b_x, stage_b_pr_mat = \
            self.input_to_correct_shape(stage_a_x, stage_a_c, stage_a_pr_mat, stage_b_x, stage_b_pr_mat)
        outputs = self.run(stage_a_x, stage_a_c, stage_a_pr_mat, stage_b_x, stage_b_pr_mat, tfr1, tfr2, tfr3)
        loss = self.loss_function(stage_a_x, stage_a_c, stage_b_x, *outputs, beta, weights)
        return loss

    @staticmethod
    def input_to_correct_shape(stage_a_x, stage_a_c, stage_a_pr_mat, stage_b_x, stage_b_pr_mat):
        stage_a_x = stage_a_x.squeeze(0)
        stage_a_c = stage_a_c.squeeze(0)
        stage_a_c = stage_a_c[:, ::4, :]
        stage_a_pr_mat = stage_a_pr_mat.squeeze(0)
        stage_b_x = stage_b_x.squeeze(0)
        stage_b_pr_mat = stage_b_pr_mat.squeeze(0)
        shape = stage_b_x.shape
        stage_b_x = stage_b_x.reshape(shape[0] * shape[1], shape[2], shape[3], shape[4])
        shape = stage_b_pr_mat.shape
        stage_b_pr_mat = stage_b_pr_mat.reshape(shape[0] * shape[1], shape[2], shape[3])
        print(stage_a_x.shape, stage_a_c.shape, stage_a_pr_mat.shape, stage_b_x.shape, stage_b_pr_mat.shape)
        return stage_a_x, stage_a_c, stage_a_pr_mat, stage_b_x, stage_b_pr_mat

    def inference(self, pr_mat, c, sample, save_z=None):
        pass

    def inference_with_chord_decode(self, pr_mat, c, sample):
        pass

    def loss_function(self, stage_a_x, stage_a_c, stage_b_x,
                      stage_a_recon_root, stage_a_recon_chroma, stage_a_recon_bass,
                      stage_a_pitch_outs, stage_a_dur_outs,
                      stage_b_pitch_outs, stage_b_dur_outs,
                      stage_a_positive, stage_a_negative, stage_a_pred,
                      stage_b_positive, stage_b_negative, stage_b_pred,
                      beta, weights, weighted_dur=False):

        chord_loss, root, chroma, bass = self.chord_loss(stage_a_c[1:], stage_a_recon_root,
                                                         stage_a_recon_chroma,
                                                         stage_a_recon_bass)
        stage_a_recon_loss, stage_a_pl, stage_a_dl = self.stage_a_decoder.recon_loss(stage_a_x[1:], stage_a_pitch_outs,
                                                                                     stage_a_dur_outs,
                                                                                     weights, weighted_dur)
        stage_b_recon_loss, stage_b_pl, stage_b_dl = self.stage_b_decoder.recon_loss(stage_b_x[5:], stage_b_pitch_outs,
                                                                                     stage_b_dur_outs, weights,
                                                                                     weighted_dur)

        stage_a_arg_loss = self.stage_a_arg_loss_fun(stage_a_pred, stage_a_positive, stage_a_negative, temperature=1)
        stage_b_arg_loss = self.stage_b_arg_loss_fun(stage_b_pred, stage_b_positive, stage_b_negative, temperature=1)
        loss = chord_loss + stage_a_recon_loss + stage_b_recon_loss + stage_a_arg_loss + stage_b_arg_loss
        return loss, chord_loss, stage_a_recon_loss, stage_a_pl, stage_a_dl, stage_b_recon_loss, stage_b_pl, \
               stage_b_dl, stage_a_arg_loss, stage_b_arg_loss

    def arg_loss(self, pred, positive, negative, temperature=1):
        return self.arg_loss_fun(pred, positive, negative, temperature)

    def chord_loss(self, c, recon_root, recon_chroma, recon_bass):
        loss_fun = nn.CrossEntropyLoss()
        root = c[:, :, 0: 12].max(-1)[-1].view(-1).contiguous()
        chroma = c[:, :, 12: 24].long().view(-1).contiguous()
        bass = c[:, :, 24:].max(-1)[-1].view(-1).contiguous()

        recon_root = recon_root.view(-1, 12).contiguous()
        recon_chroma = recon_chroma.view(-1, 2).contiguous()
        recon_bass = recon_bass.view(-1, 12).contiguous()
        root_loss = loss_fun(recon_root, root)
        chroma_loss = loss_fun(recon_chroma, chroma)
        bass_loss = loss_fun(recon_bass, bass)
        chord_loss = root_loss + chroma_loss + bass_loss
        return chord_loss, root_loss, chroma_loss, bass_loss

    @staticmethod
    def init_model(device=None, chd_size=256, voicing_size=256, txt_size=256, num_channel=10):

        name = 'disvae2'
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available()
                                  else 'cpu')

        stage_a_chd_encoder = RnnEncoder(36, 1024, chd_size)
        stage_a_rhy_encoder = TextureEncoder(256, 1024, txt_size, num_channel)
        stage_a_chd_decoder = RnnDecoder(z_dim=chd_size)
        stage_a_pt_decoder = PtvaeDecoder(note_embedding=None,
                                          dec_dur_hid_size=64,
                                          z_size=chd_size + txt_size)
        stage_a_arg_decoder = zTransformer(dim_model=512, num_heads=8, num_decoder_layers=12, dropout_p=0.1)
        stage_a_arg_loss_fun = InfoNCELoss(input_dim=512, sample_dim=512, skip_projection=False)

        stage_b_voicing_encoder = TextureEncoder(256, 1024, 256)
        stage_b_rhy_encoder = TextureEncoder(256, 1024, 256)
        stage_b_voicing_decoder = PtvaeDecoder(note_embedding=None,
                                               dec_dur_hid_size=64, z_size=256)
        stage_b_pt_decoder = PtvaeDecoder(note_embedding=None,
                                          dec_dur_hid_size=64, z_size=512)
        stage_b_arg_decoder = zTransformer(dim_model=512, num_heads=8, num_decoder_layers=12, dropout_p=0.1)
        stage_b_arg_loss = InfoNCELoss(input_dim=512, sample_dim=512, skip_projection=False)

        model = DisentangleARGFull(name, device, stage_a_chd_encoder, stage_a_rhy_encoder, stage_a_chd_decoder,
                                   stage_a_pt_decoder, stage_a_arg_decoder, stage_a_arg_loss_fun,
                                   stage_b_voicing_encoder, stage_b_rhy_encoder, stage_b_voicing_decoder,
                                   stage_b_pt_decoder, stage_b_arg_decoder, stage_b_arg_loss)
        return model

    def run(self, stage_a_x, stage_a_c, stage_a_pr_mat, stage_b_x, stage_b_pr_mat,
            tfr1, tfr2, tfr3, confuse=True):

        dist_chd = self.stage_a_chd_encoder(stage_a_c)
        dist_rhy = self.stage_a_rhy_encoder(stage_a_pr_mat)
        z_chd, z_rhy = get_zs_from_dists([dist_chd, dist_rhy], False)
        dec_z = torch.cat([z_chd, z_rhy], dim=-1).unsqueeze(0)

        y_input = dec_z[:, :-1]
        y_expected = dec_z[:, 1:]
        stage_a_pred = self.stage_a_arg_decoder(y_input)
        stage_a_pred = stage_a_pred[0]
        stage_a_positive = torch.permute(y_expected, (1, 0, 2))
        stage_a_negative = torch.concat(
            [torch.unsqueeze(torch.cat((stage_a_positive[: i, 0], stage_a_positive[i + 1:, 0]), dim=0), dim=0) \
             for i in range(stage_a_positive.shape[0])], dim=0)

        embedded_x, lengths = self.stage_a_decoder.emb_x(stage_a_x[1:])

        stage_a_pitch_outs, stage_a_dur_outs = self.stage_a_decoder(stage_a_pred, False, embedded_x, lengths, tfr1,
                                                                    tfr2)
        stage_a_recon_root, stage_a_recon_chroma, stage_a_recon_bass = self.stage_a_chd_decoder(stage_a_pred[:, 0:256],
                                                                                                False, tfr3,
                                                                                                stage_a_c[1:])

        est_pitch = stage_a_pitch_outs.max(-1)[1].unsqueeze(-1)
        est_dur = stage_a_dur_outs.max(-1)[1]
        est_x = torch.cat([est_pitch, est_dur], dim=-1)
        est_x = est_x.cpu().detach().numpy()
        all_prs = []
        for idx in range(est_x.shape[0]):
            pr, _ = self.stage_a_decoder.grid_to_pr_and_notes(grid=est_x[idx], bpm=120, start=0)
            stretched_pr = np.zeros((4, 32, 128))
            for i in range(32):
                for j in range(128):
                    stretched_pr[i * 4 // 32, i * 4 % 32, j] = pr[i, j] * 4
            all_prs.append(stretched_pr)

        stage_b_pr_mat_c = torch.tensor(all_prs).to(self.device).float()
        shape = stage_b_pr_mat_c.shape
        stage_b_pr_mat_c = stage_b_pr_mat_c.reshape(shape[0] * shape[1], shape[2], shape[3])

        embedded_x, lengths = self.stage_b_decoder.emb_x(stage_b_x[5:])
        dist_voicing = self.stage_b_voicing_encoder(stage_b_pr_mat_c)
        dist_rhy = self.stage_b_rhy_encoder(stage_b_pr_mat[4:])
        z_voicing, z_rhy = get_zs_from_dists([dist_voicing, dist_rhy], False)
        dec_z = torch.cat([z_voicing, z_rhy], dim=-1).unsqueeze(0)
        y_input = dec_z[:, :-1]
        y_expected = dec_z[:, 1:]
        stage_b_pred = self.stage_b_arg_decoder(y_input)
        stage_b_pred = stage_b_pred[0]
        stage_b_positive = torch.permute(y_expected, (1, 0, 2))
        stage_b_negative = torch.concat(
            [torch.unsqueeze(torch.cat((stage_b_positive[: i, 0], stage_b_positive[i + 1:, 0]), dim=0), dim=0) \
             for i in range(stage_b_positive.shape[0])], dim=0)
        stage_b_pitch_outs, stage_b_dur_outs = self.stage_b_decoder(stage_b_pred, False, embedded_x, lengths, tfr1,
                                                                    tfr2)

        return stage_a_recon_root, stage_a_recon_chroma, stage_a_recon_bass, \
               stage_a_pitch_outs, stage_a_dur_outs, \
               stage_b_pitch_outs, stage_b_dur_outs, \
               stage_a_positive, stage_a_negative, stage_a_pred, \
               stage_b_positive, stage_b_negative, stage_b_pred
