import numpy as np
import pretty_midi
import torch
import pretty_midi as pyd

from model import DisentangleVAE, DisentangleVoicingTextureVAE
from ptvae import PtvaeDecoder
from format_converter import chord_data2matrix, midi2pr, melody_split, chord_split, accompany_matrix2data, \
    chord_stretch, pr_stretch

np.set_printoptions(threshold=10000)


def inference_stage1(chord_table, acc_ensemble, checkpoint='data/model_master_final.pt'):
    acc_ensemble = melody_split(acc_ensemble, window_size=32, hop_size=32, vector_size=128)
    chord_table = chord_split(chord_table, 8, 8)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = DisentangleVAE.init_model(device).to(device)
    checkpoint = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    pr_matrix = torch.from_numpy(acc_ensemble).float().to(device)
    gt_chord = torch.from_numpy(chord_table).float().to(device)
    est_x, loss = model.inference_with_loss(pr_matrix, gt_chord, sample=False)
    midi_re_gen = accompaniment_generation(est_x, 30)
    return midi_re_gen


def inference_stage2(voicing, acc, checkpoint):
    acc = melody_split(acc, window_size=32, hop_size=32, vector_size=128)
    voicing = melody_split(voicing, window_size=32, hop_size=32, vector_size=128)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = DisentangleVoicingTextureVAE.init_model(device).to(device)
    checkpoint = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    acc = torch.from_numpy(acc).float().to(device)
    voicing = torch.from_numpy(voicing).float().to(device)
    est_x, loss = model.inference(acc, voicing, sample=False)
    midi_re_gen = accompaniment_generation(est_x, 120)
    return midi_re_gen


def accompaniment_generation(pr_matrix, tempo=120):
    # print(piano_roll.shape, type(piano_roll))
    pt_decoder = PtvaeDecoder(note_embedding=None, dec_dur_hid_size=64, z_size=512)
    start = 0
    tempo = tempo
    midi_re_gen = pyd.PrettyMIDI(initial_tempo=120)
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


def inference_chord_voicing_disentanglement(c_path, v_path=None, checkpoint='data/train_20220806.pt'):
    midi = pyd.PrettyMIDI(c_path)
    c = chord_data2matrix(midi.instruments[0], midi.get_downbeats(), 'quarter')
    c = c[::16, :]
    if v_path:
        v_midi = pyd.PrettyMIDI(v_path)
        v = midi2pr(v_midi.instruments[0], down_sample=4)
    else:
        v = midi2pr(midi.instruments[0], down_sample=4)
    if c.shape[0] % 2 != 0:
        c = np.concatenate((c, np.array([-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])))
    if v.shape[0] % 8 != 0:
        v = np.concatenate((v, np.zeros((8 - v.shape[0] % 8, 128))))
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
    return inference_stage1(c, v, checkpoint)


def inference_chord_voicing_texture_disentanglement(chord_provider: str,
                                                    voicing_provider: str,
                                                    texture_provider: str,
                                                    stage1_checkpoint: str,
                                                    stage2_checkpoint: str) -> pretty_midi.PrettyMIDI:
    pass


if __name__ == '__main__':
    CHORD_PATH = ''
    VOICING_PATH = ''
    TEXTURE_PATH = ''
    STAGE1_CP = ''
    STAGE2_CP = ''
    WRITE_PATH = ''
    inference_chord_voicing_texture_disentanglement(chord_provider=CHORD_PATH,
                                                    voicing_provider=VOICING_PATH,
                                                    texture_provider=TEXTURE_PATH,
                                                    stage1_checkpoint=STAGE1_CP,
                                                    stage2_checkpoint=STAGE2_CP).write(WRITE_PATH)
