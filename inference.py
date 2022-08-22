import numpy as np
import torch
import pretty_midi as pyd
from model import DisentangleVAE
from ptvae import PtvaeDecoder
import os

from format_converter import chord_data2matrix, midi2pr, melody_split, chord_split, accompany_matrix2data, \
    chord_stretch, pr_stretch


def inference(chord_table, acc_ensemble, checkpoint='model_master_final.pt'):
    acc_ensemble = melody_split(acc_ensemble, window_size=32, hop_size=32, vector_size=128)
    chord_table = chord_split(chord_table, 8, 8)
    if torch.cuda.is_available():
        model = DisentangleVAE.init_model(torch.device('cuda')).cuda()
        checkpoint = torch.load(f'data/{checkpoint}')
        model.load_state_dict(checkpoint)
        pr_matrix = torch.from_numpy(acc_ensemble).float().cuda()
        # pr_matrix_shifted = torch.from_numpy(pr_matrix_shifted).float().cuda()
        gt_chord = torch.from_numpy(chord_table).float().cuda()
        # print(gt_chord.shape, pr_matrix.shape)
        est_x, loss = model.inference_with_loss(pr_matrix, gt_chord, sample=False)
        print(int(loss[1]))
        # print('est:', est_x.shape)
        # est_x_shifted = model.inference(pr_matrix_shifted, gt_chord, sample=False)
        midi_re_gen = accompaniment_generation(est_x, 30)
        return midi_re_gen
        # midiReGen.write('accompaniment_test_NEW.mid')
    else:
        model = DisentangleVAE.init_model(torch.device('cpu'))
        checkpoint = torch.load(f'data/{checkpoint}', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        pr_matrix = torch.from_numpy(acc_ensemble).float()
        gt_chord = torch.from_numpy(chord_table).float()
        est_x, loss = model.inference_with_loss(pr_matrix, gt_chord, sample=False)
        # print(format((1 - loss[1]) * 100, '.3f') + '%')
        midi_re_gen = accompaniment_generation(est_x, 30)
        return midi_re_gen


def accompaniment_generation(pr_matrix, tempo=120):
    # print(piano_roll.shape, type(piano_roll))
    pt_decoder = PtvaeDecoder(note_embedding=None, dec_dur_hid_size=64, z_size=512)
    start = 0
    tempo = tempo
    midi_re_gen = pyd.PrettyMIDI(initial_tempo=tempo)
    texture_track = pyd.Instrument(program=pyd.instrument_name_to_program('Acoustic Grand Piano'))
    for idx in range(0, pr_matrix.shape[0]):
        if pr_matrix.shape[-1] == 6:
            pr, _ = pt_decoder.grid_to_pr_and_notes(grid=pr_matrix[idx], bpm=tempo, start=0)
        else:
            pr = pr_matrix[idx]
        # print(pr.shape)
        texture_notes = accompany_matrix2data(pr_matrix=pr, tempo=tempo, start_time=start, get_list=True)
        texture_track.notes += texture_notes
        start += 60 / tempo * 8
    midi_re_gen.instruments.append(texture_track)
    return midi_re_gen


def inference_voicing_disentanglement_8_bars_segment(path):
    midi = pyd.PrettyMIDI(path)
    c = chord_data2matrix(midi.instruments[0], midi.get_downbeats(), 'quarter')
    c = c[::16, :]
    v = midi2pr(midi.instruments[0], down_sample=4)
    print(c.shape, v.shape)
    assert c.shape[0] * 4 == v.shape[0]
    if c.shape[0] % 8 != 0:
        if c.shape[0] % 4 == 0:
            c = np.concatenate((c[:-4, :], chord_stretch(c[-4:, :], 2)))
            v = np.concatenate((v[:-16, :], pr_stretch(v[-16:, :], 2)))
        else:
            assert c.shape[0] % 2 == 0
            if c.shape[0] % 8 == 2:
                c = np.concatenate((c[:-2, :], chord_stretch(c[-2:, :], 4)))
                v = np.concatenate((v[:-8, :], pr_stretch(v[-8:, :], 4)))
            else:
                c = np.concatenate((c[:-2, :], chord_stretch(c[-2:, :], 2)))
                v = np.concatenate((v[:-8, :], pr_stretch(v[-8:, :], 2)))
    print(c.shape, v.shape)
    return inference(c, v, 'train_20220818.pt')


if __name__ == '__main__':
    for filename in os.listdir('test'):
        idx = int(filename.split('.')[0])
        print(idx)
        recon = inference_voicing_disentanglement_8_bars_segment(os.path.join('test', filename))
        input()
