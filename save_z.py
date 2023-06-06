import os

import numpy as np
import pretty_midi
import torch
from tqdm import tqdm

from inference import compute_voicing_multihot, accompaniment_generation
from models.model import DisentangleVAE, DisentangleVoicingTextureVAE
from utils.utils import melody_split, chord_split, extract_voicing, chord_data2matrix, midi2pr, pr2midi


def inference_stage1(chord_table, acc_ensemble, checkpoint='data/model_master_final.pt', save_z=None, decode_z=None):
    if decode_z is not None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = DisentangleVAE.init_model(device).to(device)
        checkpoint = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint)
        return model.inference_only_decode(decode_z)

    acc_ensemble = melody_split(acc_ensemble, window_size=32, hop_size=32, vector_size=128)
    chord_table = chord_split(chord_table, 8, 8)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = DisentangleVAE.init_model(device).to(device)
    checkpoint = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    pr_matrix = torch.from_numpy(acc_ensemble).float().to(device)
    gt_chord = torch.from_numpy(chord_table).float().to(device)
    if save_z:
        model.inference_save_z(pr_matrix, gt_chord, sample=False, z_path=save_z)
    else:
        return model.inference_with_loss(pr_matrix, gt_chord, sample=False)


def inference_stage1_chord(chord_table, acc_ensemble, checkpoint='data/model_master_final.pt', decode_z=None):
    if decode_z is not None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = DisentangleVAE.init_model(device).to(device)
        checkpoint = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint)
        est_x, recon_root, recon_chroma, recon_bass = model.inference_only_decode(decode_z, with_chord=True)
        return recon_root, recon_chroma, recon_bass

    acc_ensemble = melody_split(acc_ensemble, window_size=32, hop_size=32, vector_size=128)
    chord_table = chord_split(chord_table, 8, 8)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = DisentangleVAE.init_model(device).to(device)
    checkpoint = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    pr_matrix = torch.from_numpy(acc_ensemble).float().to(device)
    gt_chord = torch.from_numpy(chord_table).float().to(device)
    return model.inference_with_loss(pr_matrix, gt_chord, sample=False)


def chroma2midi(chroma, bass):
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    for time in range(len(chroma)):
        for pitch in range(12):
            if chroma[time, pitch][0] < chroma[time, pitch][1]:
                note = pretty_midi.Note(
                    velocity=100, pitch=pitch + 60, start=time * 2, end=(time + 1) * 2)
                piano.notes.append(note)
    for time in range(len(bass)):
        pitch = np.argmax(bass[time])
        note = pretty_midi.Note(velocity=100, pitch=pitch + 36, start=time * 2, end=(time + 1) * 2)
        piano.notes.append(note)
    midi.instruments.append(piano)
    return midi


if __name__ == '__main__':
    # for file in tqdm(os.listdir('zv')):
    #     try:
    #         chord_midi = pretty_midi.PrettyMIDI(f'zv/{file}')
    #         voicing_midi = pretty_midi.PrettyMIDI(f'zv/{file}')
    #         chord = chord_data2matrix(chord_midi.instruments[0], chord_midi.get_downbeats(), 'quarter')
    #         chord = chord[::16, :]
    #         voicing = midi2pr(voicing_midi, down_sample=4)
    #         for row in range(len(voicing)):
    #             for pitch in range(128):
    #                 if voicing[row, pitch] > 0:
    #                     voicing[row, pitch] = 4
    #         while len(chord) < 8:
    #             chord = np.row_stack((chord, np.array([0]*36)))
    #         while len(voicing) < 32:
    #             voicing = np.row_stack((voicing, np.array([0]*128)))
    #         name = file.split('.')[0]+'.pt'
    #         inference_stage1(chord, voicing, checkpoint='data/train_stage1_20220818.pt', save_z=f'zs/s1/{name}')
    #     except:
    #         print(file)

    # path = f'zv/4-23.mid_v.mid'
    # chord_midi = pretty_midi.PrettyMIDI(path)
    # voicing_midi = pretty_midi.PrettyMIDI(path)
    # chord = chord_data2matrix(chord_midi.instruments[0], chord_midi.get_downbeats(), 'quarter')
    # chord = chord[::16, :]
    # voicing = midi2pr(voicing_midi, down_sample=4)
    # for row in range(len(voicing)):
    #     for pitch in range(128):
    #         if voicing[row, pitch] > 0:
    #             voicing[row, pitch] = 4
    # while len(chord) < 8:
    #     chord = np.row_stack((chord, np.array([0]*36)))
    # while len(voicing) < 32:
    #     voicing = np.row_stack((voicing, np.array([0]*128)))
    # pr2midi(voicing).write('inspect.mid')
    # inference_stage1(chord, voicing, checkpoint='data/train_stage1_20220818.pt', save_z='hahaha.pt')
    # for folder in os.listdir('infer_output/ouput_0523'):
    #     for file in os.listdir(f'infer_output/ouput_0523/{folder}'):
    #         if file.endswith('.pt') and 'output' in file:
    #             z = torch.load(f'infer_output/ouput_0523/{folder}/{file}', map_location=torch.device('cuda'))
    #             for i in range(len(z)):
    #                 print(file, i)
    #                 # recon_root, recon_chroma, recon_bass = inference_stage1_chord(None, None,
    #                 #                                                               checkpoint='data/train_stage1_20220818.pt',
    #                 #                                                               decode_z=z[i].unsqueeze(0))
    #                 # recon_chroma = recon_chroma.squeeze(0).cpu().detach().numpy()
    #                 # recon_root = recon_root.squeeze(0).cpu().detach().numpy()
    #                 # chroma2midi(recon_chroma, recon_root).write(f'infer_output/ouput_0523/{folder}/{file.split(".")[0]}_{i}_chord.mid')
    #                 est_x = inference_stage1(None, None, checkpoint='data/train_stage1_20220818.pt', save_z=None,
    #                                          decode_z=z[i].unsqueeze(0))
    #                 accompaniment_generation(est_x, 30).write(f'infer_output/ouput_0523/{folder}/{file.split(".")[0]}_{i}.mid')
    z = torch.load(r'D:\projects\polydis2\zs\s1\487-23.pt', map_location=torch.device('cuda'))
    est_x = inference_stage1(None, None, checkpoint='data/train_stage1_20220818.pt', save_z=None, decode_z=z)
    accompaniment_generation(est_x, 30).write('487-23.mid')
    z = torch.load(r'D:\projects\polydis2\zs\s1\487-31.pt', map_location=torch.device('cuda'))
    est_x = inference_stage1(None, None, checkpoint='data/train_stage1_20220818.pt', save_z=None,
                             decode_z=z)
    accompaniment_generation(est_x, 30).write('487-31.mid')