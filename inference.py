import numpy as np
import pretty_midi
import torch
import pretty_midi as pyd
from tqdm import tqdm

from models.model import DisentangleVAE, DisentangleVoicingTextureVAE, DisentangleARG
from models.ptvae import PtvaeDecoder
from utils.utils import chord_data2matrix, midi2pr, melody_split, chord_split, accompany_matrix2data, \
    chord_stretch, pr_stretch, generate_pop909_test_sample, extract_voicing_from_pr, pr2midi, extract_voicing
from utils.utils import extract_voicing
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


def compute_voicing_multihot(voicing):
    result = []
    for item in voicing:
        i = 0
        single_result = []
        while i * 16 < len(item):
            hot = []
            for pitch in range(128):
                if item[i * 16][pitch] != 0:
                    hot.append(pitch)
            hot = [1 if j in hot else 0 for j in range(128)]
            for j in range(16):
                single_result.append(hot)
            i += 1
        result.append(single_result)
    return np.array(result)


def inference_stage2(voicing, acc, checkpoint, with_voicing_recon=False):
    acc = melody_split(acc, window_size=32, hop_size=32, vector_size=128)
    voicing = melody_split(voicing, window_size=32, hop_size=32, vector_size=128)
    vm = compute_voicing_multihot(voicing)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = DisentangleVoicingTextureVAE.init_model(device).to(device)
    checkpoint = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    acc = torch.from_numpy(acc).float().to(device)
    voicing = torch.from_numpy(voicing).float().to(device)
    vm = torch.from_numpy(vm).float().to(device)
    if with_voicing_recon:
        est_x, est_x_voicing = model.inference_with_chord_decode(acc, voicing, vm, sample=False)
        midi_re_gen = accompaniment_generation(est_x, 120)
        midi_re_gen_voicing = accompaniment_generation(est_x_voicing, 120)
        return midi_re_gen, midi_re_gen_voicing
    else:
        est_x = model.inference(acc, voicing, vm, sample=False)
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
                                                    stage2_checkpoint: str,
                                                    with_voicing_recon=False):
    if chord_provider == voicing_provider:
        voicing_midi = pretty_midi.PrettyMIDI(voicing_provider)
        texture_midi = pretty_midi.PrettyMIDI(texture_provider)
        voicing = midi2pr(voicing_midi)
        texture = midi2pr(texture_midi)
        print(texture.shape)
        print(texture)
        recon, recon_recon_voicing = inference_stage2(voicing, texture, checkpoint=stage2_checkpoint,
                                                      with_voicing_recon=with_voicing_recon)
        return recon, None, recon_recon_voicing
    else:
        chord_midi = pretty_midi.PrettyMIDI(chord_provider)
        voicing_midi = pretty_midi.PrettyMIDI(voicing_provider)
        texture_midi = pretty_midi.PrettyMIDI(texture_provider)
        chord = chord_data2matrix(chord_midi.instruments[0], chord_midi.get_downbeats(), 'quarter')
        chord = chord[::16, :]
        print(chord.shape)
        voicing = midi2pr(voicing_midi, down_sample=4)
        print(voicing.shape)
        recon_chord_voicing_midi = inference_stage1(chord, voicing, checkpoint=stage1_checkpoint)
        recon_voicing = midi2pr(recon_chord_voicing_midi)
        texture = midi2pr(texture_midi)
        print(texture.shape)
        print(texture)
        if with_voicing_recon:
            recon, recon_recon_voicing = inference_stage2(recon_voicing, texture, checkpoint=stage2_checkpoint,
                                                          with_voicing_recon=with_voicing_recon)
            return recon, recon_chord_voicing_midi, recon_recon_voicing
        else:
            recon = inference_stage2(recon_voicing, texture, checkpoint=stage2_checkpoint)
            return recon, recon_chord_voicing_midi

def inference_stage_a_arg(prompt, checkpoint):
    midi = pyd.PrettyMIDI(prompt)
    c = chord_data2matrix(midi.instruments[0], midi.get_downbeats(), 'quarter')
    c = c[::16, :]
    v = midi2pr(midi.instruments[0], down_sample=4)

    acc_ensemble = melody_split(v, window_size=32, hop_size=32, vector_size=128)
    chord_table = chord_split(c, 8, 8)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = DisentangleARG.init_model(device).to(device)
    checkpoint = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    pr_matrix = torch.from_numpy(acc_ensemble).float().to(device)
    gt_chord = torch.from_numpy(chord_table).float().to(device)
    all_est_x = model.inference(pr_matrix, gt_chord, sample=False)
    all_regen = []
    for est_x in all_est_x:
        midi_re_gen = accompaniment_generation(est_x, 30)
        all_regen.append(midi_re_gen)
    return all_regen


if __name__ == '__main__':
    # for i in range(1, 3):
    #     PATH = f'experiments/20230321/{i}/'
    #     CHORD_PATH = PATH + 'c.mid'
    #     VOICING_PATH = PATH + 'v.mid'
    #     TEXTURE_PATH = PATH + 't.mid'
    #     STAGE1_CP = 'data/train_stage1_20220818.pt'
    #     STAGE2_CP = 'data/train_stage2_20221009.pt'
    #     VOICING_WRITE_PATH = PATH + 'recon_voicing.mid'
    #     RECON_VOICING_WRITE_PATH = PATH + 'recon_recon_voicing.mid'
    #     RECON_WRITE_PATH = PATH + 'recon.mid'
    #     recon, recon_voicing, recon_recon_voicing = inference_chord_voicing_texture_disentanglement(
    #         chord_provider=CHORD_PATH,
    #         voicing_provider=VOICING_PATH,
    #         texture_provider=TEXTURE_PATH,
    #         stage1_checkpoint=STAGE1_CP,
    #         stage2_checkpoint=STAGE2_CP,
    #         with_voicing_recon=True)
    #     recon_voicing.write(VOICING_WRITE_PATH) if recon_voicing is not None else None
    #     recon.write(RECON_WRITE_PATH) if recon is not None else None
    #     recon_recon_voicing.write(RECON_VOICING_WRITE_PATH) if recon_recon_voicing is not None else None
    # print("task finished")

    # for i in range(1, 2):
    #     PATH = f'experiments/20230607/{i}/'
    #     CHORD_PATH = PATH + 'c.mid'
    #     VOICING_PATH = PATH + 'v.mid'
    #     STAGE1_CP = 'result_2023-06-06_122449/models/disvae-nozoth_final.pt'
    #     RECON_WRITE_PATH = PATH + 'recon.mid'
    #     recon = inference_chord_voicing_disentanglement(
    #         c_path=CHORD_PATH,
    #         v_path=VOICING_PATH,
    #         checkpoint=STAGE1_CP,
    #     )
    #     recon.write(RECON_WRITE_PATH) if recon is not None else None
    # print("task finished")

    # for i in tqdm(range(1, 15)):
    #     PATH = f'experiments/20230726/{i}/'
    #     recons = inference_stage_a_arg(PATH + 'p.mid', 'result_2023-07-25_173743/models/disvae-nozoth_final.pt')
    #     for j, recon in enumerate(recons):
    #         recon.write(PATH + f'recon_{j}.mid') if recon is not None else None

    PATH = f'experiments/20221009/test6/'
    CHORD_PATH = PATH + 'voicing.mid'
    VOICING_PATH = PATH + 'voicing.mid'
    TEXTURE_PATH = PATH + 'texture.mid'
    STAGE1_CP = 'data/train_stage1_20220818.pt'
    STAGE2_CP = 'data/train_stage2_20221009.pt'
    VOICING_WRITE_PATH = PATH + 'recon_voicing.mid'
    RECON_VOICING_WRITE_PATH = PATH + 'recon_recon_voicing.mid'
    RECON_WRITE_PATH = PATH + 'recon.mid'
    recon, recon_voicing, recon_recon_voicing = inference_chord_voicing_texture_disentanglement(
        chord_provider=CHORD_PATH,
        voicing_provider=VOICING_PATH,
        texture_provider=TEXTURE_PATH,
        stage1_checkpoint=STAGE1_CP,
        stage2_checkpoint=STAGE2_CP,
        with_voicing_recon=True)
    recon_voicing.write(VOICING_WRITE_PATH) if recon_voicing is not None else None
    recon.write(RECON_WRITE_PATH) if recon is not None else None
    recon_recon_voicing.write(RECON_VOICING_WRITE_PATH) if recon_recon_voicing is not None else None
