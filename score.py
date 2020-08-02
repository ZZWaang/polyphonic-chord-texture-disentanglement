from converter import ext_nmat_to_nmat, nmat_to_notes
import pretty_midi as pm
import numpy as np


class PolyphonicMusic:

    def __init__(self, tracks, beat_table, chord_table, instrument_list=None,
                 track_name_list=None, bpm=120.):
        self.tracks = tracks
        # self.beat_table = beat_table
        assert beat_table.shape[0] == chord_table.shape[0]
        self.beat_track = BeatTrack(beat_table, require_regularize=True)
        self.chord_table = chord_table  # chord table is not regularized!
        self.regularize_chord_table()
        self.regularize_tracks()
        self.num_track = len(tracks)
        if instrument_list is None:
            self.instrument_list = [0] * len(tracks)
        else:
            self.instrument_list = instrument_list
        if track_name_list is None:
            self.track_name_list = [str(i) for i in range(self.num_track)]
        else:
            self.track_name_list = track_name_list
        self.bpm = bpm

    def _select_track(self, track_ind=None, track_name=None):
        if track_ind is None and track_name is None:
            track_ind = 0
        elif track_ind is None:
            track_ind = self.track_name_list.index(track_name)
        track = self.tracks[track_ind]
        return track

    def _break_track_to_bars(self, track, db_pos=None, db_ts=None):
        # return a list of bars
        # track = self._select_track(track_ind, track_name)
        if db_pos is None or db_ts is None:
            db_pos, db_ts = self.beat_track.get_downbeats()
        bar_tracks = []
        for s, e in zip(db_pos, np.append(db_pos[1: ],
                                          db_pos[-1] + db_ts[-1])):
            note_inds = \
                np.where(np.logical_and(track[:, 0] >= s, track[:, 0] < e))[0]
            bar_track = track[note_inds]
            bar_tracks.append(bar_track)
        return bar_tracks

    def _break_chord_to_bars(self, track, db_pos=None, db_ts=None):
        if db_pos is None or db_ts is None:
            db_pos, db_ts = self.beat_track.get_downbeats()
        bar_chord = []
        for s, e in zip(db_pos, np.append(db_pos[1: ],
                                          db_pos[-1] + db_ts[-1])):
            bar_chord.append(self.chord_table[s: e])
        return bar_chord

    def break_tracks_to_bars(self, db_pos=None, db_ts=None):
        if db_pos is None or db_ts is None:
            db_pos, db_ts = self.beat_track.get_downbeats()
        broken_tracks = \
            [self._break_track_to_bars(track, db_pos, db_ts)
             for track in self.tracks]
        broken_tracks = [list(bar) for bar in zip(*broken_tracks)]
        return broken_tracks

    def prepare_data(self, num_bar=8, ts=4, mel_id=(0,), acc_id=(1, 2)):
        # Indicator == 1 if
        # 1) The current bar has ts = 4
        # 2) The current bar is not blank
        # 3) The consecutive 3 bars have ts = 4.
        def merge(tracks, ids):
            to_merge = [track for i, track in enumerate(tracks) if i in ids and
                        len(track) > 0]

            if len(to_merge) == 0:
                return None
            else:
                merged = np.concatenate(to_merge, axis=0)
                merged = merged[merged[:, 0].argsort()]
                return merged

        def translate_track(track, translation):
            if track is None:
                return track
            track[:, 0] -= translation
            track[:, 3] -= translation
            return track

        db_pos, db_ts = self.beat_track.get_downbeats()
        broken_tracks = self.break_tracks_to_bars(db_pos, db_ts)
        broken_chords = self._break_chord_to_bars(db_pos, db_ts)
        assert len(broken_tracks) == len(db_pos)
        assert len(broken_chords) == len(db_pos)
        indicator = np.zeros(len(db_pos))
        # Do the following check because there are cases when ts = 4, but
        # the beat count is 4. In this case, there is only 2 beat in this
        # bar. We should not include those bars.
        for i, chord in enumerate(broken_chords):
            if chord.shape[0] != ts:
                indicator[i] = -1
        data_track = []
        for i in range(len(db_pos)):
            tracks = broken_tracks[i]
            mel_track = translate_track(merge(tracks, mel_id), db_pos[i])
            acc_track = translate_track(merge(tracks, acc_id), db_pos[i])
            chord = broken_chords[i]
            data_track.append([mel_track, acc_track, chord])
            if mel_track is None and acc_track is None:
                indicator[i] = 0
                continue
            if i > len(db_pos) - num_bar:
                indicator[i] = 0
                continue
            if not (db_ts[i: i + num_bar] == ts).all():
                indicator[i] = 0
                continue
            if (indicator[i: i + num_bar] == -1).any():
                indicator[i] = 0
                continue
            indicator[i] = 1
        return data_track, indicator, db_pos

    def regularize_chord_table(self):
        pre_cat = np.zeros((self.beat_track.translation,
                            self.chord_table.shape[1]),
                           dtype=self.chord_table.dtype)
        post_cat = np.zeros((self.beat_track.post_translation,
                             self.chord_table.shape[1]),
                            dtype=self.chord_table.dtype)
        self.chord_table = \
            np.concatenate([pre_cat, self.chord_table, post_cat], axis=0)

    def regularize_track(self, track):
        track[:, 0] += self.beat_track.translation
        track[:, 3] += self.beat_track.translation
        return track

    def regularize_tracks(self):
        self.tracks = [self.regularize_track(track) for track in self.tracks]

    def convert_track_to_nmat(self, track_ind=None, track_name=None):
        track = self._select_track(track_ind, track_name)
        return ext_nmat_to_nmat(track)

    def convert_track_to_notes(self, track_ind=None, track_name=None,
                               start=0., bpm=None):
        if bpm is None:
            bpm = self.bpm
        nmat = self.convert_track_to_nmat(track_ind, track_name)
        notes = nmat_to_notes(nmat, start, bpm)
        return notes

    def convert_tracks(self, track_ids=None, track_names=None,
                       start=0., bpm=None):
        if bpm is None:
            bpm = self.bpm
        if track_ids is None:
            track_ids = [None] * self.num_track
        if track_names is None:
            track_names = [None] * self.num_track
        track_notes = []
        for track_id, track_name in zip(track_ids, track_names):
            notes = self.convert_track_to_notes(track_id, track_name,
                                                start, bpm)
            track_notes.append(notes)
        return track_notes

    def export_to_pretty_midi(self, track_ids=None, track_names=None,
                              start=0., bpm=None):
        if bpm is None:
            bpm = self.bpm
        if track_ids is None and track_names is None:
            track_ids = list(range(self.num_track))
            track_names = self.track_name_list
        elif track_ids is None:
            track_ids = [self.track_name_list.index(name)
                         for name in track_names]
        elif track_names is None:
            track_names = [self.track_name_list[i] for i in track_ids]
        inst_ids = [self.instrument_list[i] for i in track_ids]
        track_notes = self.convert_tracks(track_ids, track_names, start, bpm)

        midi = pm.PrettyMIDI()
        for notes, inst, name in zip(track_notes, inst_ids, track_names):
            instrument = pm.Instrument(inst, name=name)
            instrument.notes = notes
            midi.instruments.append(instrument)
        return midi

    def write_midi(self, fn, track_ids=None, track_names=None,
                   start=0., bpm=None):
        midi = self.export_to_pretty_midi(track_ids, track_names,
                                          start, bpm)
        midi.write(fn)


class BeatTrack:

    def __init__(self, beat_table, require_regularize=True):
        self.beat_table = beat_table
        self.translation = 0
        self.post_translation = 0
        if require_regularize:
            self.regularize_beat_table()
        self.regularized_track = self._is_regularized_track()

    def _is_regularized_track(self):
        return self._is_pre_regularized_track() and \
               self._is_post_regularized_track()

    def _is_pre_regularized_track(self):
        return self.beat_table[0, 3] == 0

    def _is_post_regularized_track(self):
        return self.beat_table[-1, 3] == self.beat_table[-1, 5] - 1

    def _fill_pre_beat(self):
        """Add a beat row to the previous table"""
        cur_beat = self.beat_table[0]
        pre_beat = np.copy(cur_beat)
        pre_beat[0] = (pre_beat[0] - 1) % pre_beat[2]
        pre_beat[3] = (pre_beat[3] - 1) % pre_beat[5]
        if cur_beat[0] == 0:
            pre_beat[1] -= 1
        if cur_beat[3] == 0:
            pre_beat[4] -= 1
        pre_beat = np.expand_dims(pre_beat, 0)
        self.beat_table = np.concatenate([pre_beat, self.beat_table], axis=0)

    def _fill_post_beat(self):
        cur_beat = self.beat_table[0]
        post_beat = np.copy(cur_beat)
        post_beat[0] = (post_beat[0] + 1) % post_beat[2]
        post_beat[3] = (post_beat[3] + 1) % post_beat[5]
        if post_beat[0] == 0:
            post_beat[1] += 1
        if post_beat[3] == 0:
            post_beat[4] += 1
        post_beat = np.expand_dims(post_beat, 0)
        self.beat_table = np.concatenate([self.beat_table, post_beat], axis=0)

    def _renumber_beat(self):
        self.beat_table[:, 1] -= self.beat_table[0, 1]
        self.beat_table[:, 4] -= self.beat_table[0, 4]

    def regularize_beat_table(self):
        while not self.beat_table[0, 3] == 0:
            self._fill_pre_beat()
            self.translation += 1
        while not self.beat_table[-1, 3] != self.beat_table[-1, 5] - 1:
            self._fill_post_beat()
            self.post_translation += 1

    def get_downbeats(self):
        db_pos = np.where(self.beat_table[:, 3] == 0)[0]
        db_ts = self.beat_table[db_pos, 5]
        return db_pos, db_ts

    def get_time_signature_change(self):
        if not self.regularized_track:
            raise AssertionError("Beat track should be regularized first.")
        beat_info = self.beat_table[:, 5]
        ts_change_pos = \
            np.concatenate([np.zeros(1, dtype=int),
                            np.where(beat_info[1:] !=
                                     np.roll(beat_info, 1)[1:])[0] + 1])
        ts_values = self.beat_table[ts_change_pos, 5]
        return ts_change_pos, ts_values

