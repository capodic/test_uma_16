# ------------------------------------------------------------------------------
# UMA-16 Drone Locator - applicazione Bokeh basata su bf_example_app (spectacoular)
# ------------------------------------------------------------------------------
"""
Localizzazione e tracciamento in tempo reale di sorgenti acustiche (droni) con l'array
miniDSP UMA-16 (16 microfoni MEMS, griglia 4x4, passo 42 mm).

Avvio:
    python main.py                 (apre il browser su http://localhost:5006/)
oppure
    bokeh serve --show main.py
"""

from __future__ import annotations

import base64
import io
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

os.environ.setdefault('SD_ENABLE_ASIO', '1')
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["OPENBLAS_NUM_THREADS"] = "1"

try:  # acoular va importato prima di numpy (parallelismo numba)
    import acoular  # noqa: F401,E402
except Exception:
    acoular = None
from time import sleep
sleep(10)
import numpy as np  # noqa: E402
from bokeh.io import curdoc  # noqa: E402
from bokeh.layouts import column, row  # noqa: E402
from bokeh.models import (  # noqa: E402
    BoxAnnotation, Button, CheckboxGroup, ColorBar, ColumnDataSource, Div, FileInput, HoverTool,
    LabelSet, LinearColorMapper, MultiChoice, PreText, Range1d, RangeSlider, Select, Slider, Spacer,
    Spinner, TabPanel, Tabs, TextInput, Toggle,
)
from bokeh.palettes import Category20, Turbo256, viridis  # noqa: E402
from bokeh.plotting import figure  # noqa: E402
from bokeh.server.server import Server  # noqa: E402

import audio_io  # noqa: E402
from calibration import Calibration, check_channels, find_orientation, gain_calibration, phase_calibration  # noqa: E402
from config import DATA_DIR, DEFAULT_CONFIG_FILE, ENVIRONMENT_PRESETS, AppConfig  # noqa: E402
from geometry import available_geometries, load_positions  # noqa: E402
from noise import NoiseModel, noise_file  # noqa: E402
from processing import METHODS, Processor  # noqa: E402
from signature import SignatureModel, model_file  # noqa: E402
from tracking import Tracker  # noqa: E402

REC_DIR = DATA_DIR / 'recordings'
REC_DIR.mkdir(exist_ok=True)
CALIB_FILE = DATA_DIR / 'calibration.json'
SOURCES = ['UMA-16 (live)', 'File (h5/wav)', 'Simulatore']
SPEC_MODES = ['Somma (media canali)', 'Focalizzato sulla sorgente', 'Tutti i canali'] + \
             [f'Canale {i}' for i in range(16)]
CLASS_PRESETS = ['drone', 'sfondo', 'uccelli', 'traffico', 'vento', 'voci', 'altro']
EPS = 1e-20
SPEC_FRAMES = 80
CH_COLORS = list(Category20[20])[:16]


def db(x):
    return 10 * np.log10(np.asarray(x) + EPS)


def sp(title, value, step, low=None, high=None, width=110):
    return Spinner(title=title, value=value, step=step, low=low, high=high, width=width)


def checkbox(label, active):
    return CheckboxGroup(labels=[label], active=[0] if active else [])


def cb_val(w):
    return 0 in w.active


def horizon_lines():
    """Cerchi di elevazione (0, 30, 60 gradi) e raggi di azimut per la mappa (u,v)."""
    xs, ys = [], []
    t = np.linspace(0, 2 * np.pi, 181)
    for el in (0, 30, 60):
        r = np.cos(np.radians(el))
        xs.append(list(r * np.cos(t)))
        ys.append(list(r * np.sin(t)))
    for az in range(0, 360, 45):
        xs.append([0, np.cos(np.radians(az))])
        ys.append([0, np.sin(np.radians(az))])
    return dict(xs=xs, ys=ys)


# ============================================================================== applicazione
class DroneApp:
    def __init__(self, doc):
        self.doc = doc
        self.cfg = AppConfig.from_json()
        self.calib = Calibration()
        if CALIB_FILE.exists():
            try:
                self.calib.load(CALIB_FILE)
            except Exception:
                pass
        self.noise = NoiseModel()
        if noise_file().exists():
            try:
                self.noise.load(noise_file())
            except Exception:
                pass
        self.model = SignatureModel()
        if model_file().exists():
            try:
                self.model.load(model_file())
            except Exception:
                pass
        self.proc = Processor(self.cfg, self.calib, self.noise)
        self.tracker = Tracker(self.cfg.track_gate, self.cfg.track_process_noise, max_missed=self.cfg.track_max_missed)
        self.source = None
        self.recorder = None
        self.buf = np.zeros((0, 16), np.float32)
        self.collect = None
        self.last = None
        self.t0 = time.time()
        self.spec_img = np.full((len(self.proc.freqs), SPEC_FRAMES), -120.0)
        self.cb_id = None
        self.proc_ms = 0.0
        self._update_signature_weights()

        self._build_live()
        self._build_trajectory()
        self._build_calibration()
        self._build_noise()
        self._build_training()
        self._build_params()
        self._build_offline()
        self._build_guide()
        self.tabs = Tabs(tabs=[
            TabPanel(child=self.live_layout, title='Live'),
            TabPanel(child=self.traj_layout, title='Traiettoria'),
            TabPanel(child=self.cal_layout, title='Calibrazione'),
            TabPanel(child=self.noise_layout, title='Rumore / Ambiente'),
            TabPanel(child=self.train_layout, title='Addestramento'),
            TabPanel(child=self.param_layout, title='Parametri'),
            TabPanel(child=self.off_layout, title='Analisi offline (Acoular)'),
            TabPanel(child=self.guide_layout, title='Guida'),
        ])
        header = Div(text='<h2 style="margin:4px 0">UMA-16 Drone Locator</h2>'
                          '<span style="color:#666">Beamforming in tempo reale, spettro, traiettoria, '
                          'calibrazione, riduzione del rumore e riconoscimento dell\'impronta acustica</span>')
        self.root = column(header, self.tabs, sizing_mode='stretch_width')
        doc.add_root(self.root)
        doc.title = 'UMA-16 Drone Locator'
        doc.on_session_destroyed(self._on_destroy)
        self._refresh_grid_view()

    # ------------------------------------------------------------------ utilita'
    def status(self, msg, err=False):
        color = '#b00020' if err else '#1b5e20'
        self.status_div.text = f'<span style="color:{color}">{msg}</span>'

    def _on_destroy(self, _ctx):
        self.stop_source()

    def _update_signature_weights(self):
        self.proc.signature_weights = self.model.frequency_weights(self.proc.freqs) if self.model.trained else None

    def rebuild(self):
        """Ricostruisce il processore (griglia, steering) dopo cambi di parametri."""
        try:
            self.proc = Processor(self.cfg, self.calib, self.noise)
            self._update_signature_weights()
            gsig = (self.cfg.grid_mode, self.cfg.x_min, self.cfg.x_max, self.cfg.y_min, self.cfg.y_max, self.cfg.z)
            if gsig != getattr(self, '_grid_sig', gsig):   # coordinate cambiate: la traccia non e' piu' valida
                self._clear_track()
            self._grid_sig = gsig
            self._set_tracker('track_gate', self.cfg.track_gate)
            self.spec_img = np.full((len(self.proc.freqs), SPEC_FRAMES), -120.0)
            self._refresh_grid_view()
        except Exception as exc:
            self.status(f'Errore parametri: {exc}', err=True)
            traceback.print_exc()

    # ================================================================== TAB LIVE
    def _build_live(self):
        cfg = self.cfg
        # --- controlli
        self.src_select = Select(title='Sorgente segnale', options=SOURCES, value=cfg.source, width=170)
        self.run_toggle = Toggle(label='▶ Avvia acquisizione', button_type='success', width=190, height=50)
        self.run_toggle.on_change('active', self._on_run)
        self.rec_toggle = Toggle(label='● Registra (h5)', button_type='danger', width=140, height=50)
        self.rec_toggle.on_change('active', self._on_record)
        self.method_select = Select(title='Metodo beamforming', options=METHODS, value=cfg.method, width=170)
        self.method_select.on_change('value', lambda a, o, n: self._set('method', n, rebuild=False))
        self.band_slider = RangeSlider(title='Banda di analisi (Hz)', start=100, end=cfg.sample_rate / 2 * 0.95,
                                       step=50, value=(cfg.f_min, cfg.f_max), width=330)
        self.band_slider.on_change('value_throttled', self._on_band)
        self.dyn_slider = Slider(title='Dinamica mappa (dB sotto il picco)', start=1, end=30, step=0.5, value=6,
                                 width=230)
        self.dyn_slider.on_change('value', lambda a, o, n: setattr(self.cmap, 'low', -n))
        self.thr_slider = Slider(title='Soglia direzionalità (dB)', start=0, end=30, step=0.5,
                                 value=cfg.detect_threshold_db, width=230)
        self.thr_slider.on_change('value', lambda a, o, n: self._set('detect_threshold_db', n, rebuild=False))
        self.snr_slider = Slider(title='Soglia SNR vs fondo (dB)', start=-5, end=30, step=0.5,
                                 value=cfg.snr_detect_db, width=230)
        self.snr_slider.on_change('value', lambda a, o, n: self._set('snr_detect_db', n, rebuild=False))
        self.clear_btn = Button(label='Azzera traiettoria', width=140)
        self.clear_btn.on_click(self._clear_track)
        self.status_div = Div(text='Pronto. Selezionare la sorgente e premere Avvia.', width=700)
        self.info_div = Div(text='', width=330)

        # --- mappa sorgente (come "Source Map" di bf_example_app)
        self.cmap = LinearColorMapper(palette=viridis(100), low=-6, high=0, low_color=(0, 0, 0, 0),
                                      nan_color=(0, 0, 0, 0))
        self.map_src = ColumnDataSource(data=dict(image=[np.full((2, 2), -100.0)], x=[-1], y=[-1], dw=[2], dh=[2]))
        self.map_fig = figure(title='Mappa sorgente', width=640, height=560, x_range=Range1d(-1, 1),
                              y_range=Range1d(-1, 1), tools='pan,wheel_zoom,reset,save')
        self.map_fig.toolbar.logo = None
        self.map_fig.xgrid.visible = self.map_fig.ygrid.visible = False
        img = self.map_fig.image(image='image', x='x', y='y', dw='dw', dh='dh', color_mapper=self.cmap,
                                 source=self.map_src, alpha=0.9)
        self.map_fig.add_layout(ColorBar(color_mapper=self.cmap, title='dB', title_standoff=8), 'right')
        self.horizon_src = ColumnDataSource(data=dict(xs=[], ys=[]))
        self.map_fig.multi_line('xs', 'ys', source=self.horizon_src, line_color='#9e9e9e', line_alpha=0.6,
                                line_dash='dotted')
        self.mic_src = ColumnDataSource(data=dict(x=[], y=[], ch=[]))
        mic_r = self.map_fig.scatter('x', 'y', source=self.mic_src, marker='circle_cross', size=12,
                                     fill_alpha=0.2, line_color='#1e3246')
        self.trail_src = ColumnDataSource(data=dict(x=[], y=[]))
        self.map_fig.line('x', 'y', source=self.trail_src, line_color='white', line_width=2, line_alpha=0.8)
        self.peak_src = ColumnDataSource(data=dict(x=[], y=[]))
        self.map_fig.scatter('x', 'y', source=self.peak_src, marker='x', size=16, line_width=3, color='red')
        self.trk_src = ColumnDataSource(data=dict(x=[], y=[], c=[]))
        self.map_fig.scatter('x', 'y', source=self.trk_src, marker='circle', size=18, fill_alpha=0,
                             line_width=3, line_color='c')
        self.true_src = ColumnDataSource(data=dict(x=[], y=[]))
        self.map_fig.scatter('x', 'y', source=self.true_src, marker='square', size=10, fill_alpha=0,
                             line_color='orange', line_width=2, legend_label='posizione vera (sim./calib.)')
        self.map_fig.legend.location = 'top_left'
        self.map_fig.legend.background_fill_alpha = 0.3
        self.map_fig.add_tools(HoverTool(tooltips=[('dB', '@image{0.0}')], renderers=[img]))
        self.map_fig.add_tools(HoverTool(tooltips=[('microfono', '@ch')], renderers=[mic_r]))

        # --- spettro (come "Sector-Integrated Spectrum" di bf_example_app)
        self.spec_mode = Select(title='Spettro', options=SPEC_MODES, value=SPEC_MODES[0], width=220)
        self.spec_fig = figure(title='Spettro di frequenza', width=640, height=300, x_axis_type='log',
                               x_axis_label='f / Hz', y_axis_label='livello / dB (rel.)',
                               tools='pan,wheel_zoom,box_zoom,reset,save')
        self.spec_fig.toolbar.logo = None
        self.spec_fig.x_range = Range1d(50, cfg.sample_rate / 2)
        self.spec_fig.y_range = Range1d(-120, 0)
        self.spec_lines = ColumnDataSource(data=dict(xs=[], ys=[], color=[], label=[]))
        self.spec_fig.multi_line('xs', 'ys', color='color', source=self.spec_lines, line_width=1.5)
        self.spec_noise = ColumnDataSource(data=dict(x=[], y=[]))
        self.spec_fig.line('x', 'y', source=self.spec_noise, color='gray', line_dash='dashed',
                           legend_label='profilo rumore')
        self.spec_fig.legend.location = 'bottom_left'
        self.band_box = BoxAnnotation(left=cfg.f_min, right=cfg.f_max, fill_alpha=0.07, fill_color='green')
        self.spec_fig.add_layout(self.band_box)
        self.spl_slider = RangeSlider(title='Asse livello (dB)', start=-160, end=40, step=1, value=(-120, 0),
                                      width=250)
        self.spl_slider.on_change('value', lambda a, o, n: (setattr(self.spec_fig.y_range, 'start', n[0]),
                                                           setattr(self.spec_fig.y_range, 'end', n[1])))
        # --- spettrogramma
        self.sg_map = LinearColorMapper(palette=Turbo256, low=-110, high=-40)
        self.sg_src = ColumnDataSource(data=dict(image=[self.spec_img], x=[0], y=[0], dw=[SPEC_FRAMES],
                                                 dh=[cfg.sample_rate / 2]))
        self.sg_fig = figure(title='Spettrogramma (segnale focalizzato)', width=640, height=250,
                             x_axis_label='frame', y_axis_label='f / Hz', tools='pan,wheel_zoom,reset')
        self.sg_fig.toolbar.logo = None
        self.sg_fig.y_range = Range1d(0, min(10000, cfg.sample_rate / 2))
        self.sg_fig.x_range = Range1d(0, SPEC_FRAMES)
        self.sg_fig.image(image='image', x='x', y='y', dw='dw', dh='dh', color_mapper=self.sg_map, source=self.sg_src)
        self.sg_range = RangeSlider(title='Colori spettrogramma (dB)', start=-160, end=20, step=1,
                                    value=(-110, -40), width=250)
        self.sg_range.on_change('value', lambda a, o, n: (setattr(self.sg_map, 'low', n[0]),
                                                         setattr(self.sg_map, 'high', n[1])))

        controls = column(
            row(self.src_select, self.method_select),
            row(self.run_toggle, self.rec_toggle),
            self.band_slider, self.dyn_slider, self.thr_slider, self.snr_slider, self.clear_btn,
            self.info_div,
        )
        self.live_layout = column(
            self.status_div,
            row(column(self.map_fig), column(row(self.spec_mode, self.spl_slider), self.spec_fig,
                                             self.sg_range, self.sg_fig), controls),
        )

    def _set(self, attr, value, rebuild=True):
        setattr(self.cfg, attr, value)
        if rebuild:
            self.rebuild()

    def _on_band(self, _a, _o, new):
        self.cfg.f_min, self.cfg.f_max = float(new[0]), float(new[1])
        self.band_box.left, self.band_box.right = new
        self.rebuild()

    def _clear_track(self):
        self.tracker.reset()
        self.trail_src.data = dict(x=[], y=[])
        self.traj_src.data = dict(x=[], y=[], t=[], az=[], el=[], lvl=[])

    def _refresh_grid_view(self):
        g = self.proc.grid
        x0, x1, y0, y1 = g.extent
        pad = g.step / 2
        self.map_fig.x_range.start, self.map_fig.x_range.end = x0 - pad, x1 + pad
        self.map_fig.y_range.start, self.map_fig.y_range.end = y0 - pad, y1 + pad
        uv = g.mode.startswith('Emisfero')
        self.map_fig.xaxis.axis_label = 'u = sin(θ)cos(φ)' if uv else 'x / m'
        self.map_fig.yaxis.axis_label = 'v = sin(θ)sin(φ)' if uv else 'y / m'
        self.horizon_src.data = horizon_lines() if uv else dict(xs=[], ys=[])
        pos = self.proc.pos_all
        if uv:   # l'array (13 cm) e' disegnato ingrandito al centro per riferimento dell'orientamento
            k = 0.12 / 0.063
        else:
            k = 1.0
        self.mic_src.data = dict(x=list(pos[0] * k), y=list(pos[1] * k), ch=list(range(pos.shape[1])))
        if hasattr(self, 'traj_fig'):
            for f in (self.traj_fig,):
                f.x_range.start, f.x_range.end = x0 - pad, x1 + pad
                f.y_range.start, f.y_range.end = y0 - pad, y1 + pad
            self.traj_horizon.data = dict(self.horizon_src.data)
        if hasattr(self, 'geo_src'):
            self.geo_src.data = dict(x=list(pos[0]), y=list(pos[1]), ch=[str(i) for i in range(pos.shape[1])],
                                     color=['#b00020' if i in self.cfg.invalid_channels else '#1e3246'
                                            for i in range(pos.shape[1])])

    # ================================================================== ACQUISIZIONE
    def _make_source(self):
        cfg = self.cfg
        kind = self.src_select.value
        cfg.source = kind
        if kind == 'UMA-16 (live)':
            return audio_io.LiveSource(cfg.sample_rate, cfg.num_channels, cfg.device_name)
        if kind == 'File (h5/wav)':
            if not cfg.replay_file:
                raise RuntimeError('Indicare il file da riprodurre nella scheda Parametri.')
            return audio_io.FileSource(cfg.replay_file, cfg.num_channels)
        pos = load_positions(cfg.geometry)   # il simulatore usa la geometria "vera" non ribaltata
        s = audio_io.SimulatedSource(pos, cfg.sample_rate, cfg.c, cfg.sim_bpf_hz, cfg.sim_snr_db, cfg.sim_path)
        s.drone_on = cb_val(self.sim_on)
        return s

    def _on_run(self, _a, _o, active):
        if active:
            try:
                self.source = self._make_source()
                if self.source.fs != self.cfg.sample_rate:
                    self.cfg.sample_rate = self.source.fs
                    self.rebuild()
                self.source.start()
                self.buf = np.zeros((0, self.source.nch), np.float32)
                self.cb_id = self.doc.add_periodic_callback(self.tick, int(self.cfg.update_ms))
                self.run_toggle.label = '■ Ferma acquisizione'
                self.run_toggle.button_type = 'warning'
                self.status(f'Acquisizione avviata: {self.source.info}')
            except Exception as exc:
                self.source = None
                self.run_toggle.active = False
                self.status(f'Impossibile avviare la sorgente: {exc}', err=True)
        else:
            was_running = self.source is not None
            self.stop_source()
            self.run_toggle.label = '▶ Avvia acquisizione'
            self.run_toggle.button_type = 'success'
            if was_running:
                self.status('Acquisizione fermata.')

    def stop_source(self):
        if self.cb_id is not None:
            try:
                self.doc.remove_periodic_callback(self.cb_id)
            except Exception:
                pass
            self.cb_id = None
        if self.source is not None:
            self.source.stop()
            self.source = None
        if self.recorder is not None:
            self.recorder.close()
            self.recorder = None

    def _on_record(self, _a, _o, active):
        if active:
            if self.source is None:
                self.status('Avviare prima l\'acquisizione.', err=True)
                self.rec_toggle.active = False
                return
            path = REC_DIR / f'rec_{datetime.now():%Y%m%d_%H%M%S}.h5'
            self.recorder = audio_io.Recorder(path, self.source.fs, self.source.nch)
            self.rec_toggle.label = '■ Stop registrazione'
            self.status(f'Registrazione in corso: {path}')
        else:
            if self.recorder is not None:
                p, n = self.recorder.path, self.recorder.n
                self.recorder.close()
                self.recorder = None
                self.status(f'Registrazione salvata: {p} ({n / self.cfg.sample_rate:.1f} s). '
                            'Utilizzabile come sorgente "File" o in "Analisi offline".')
                self._refresh_recordings()
            self.rec_toggle.label = '● Registra (h5)'

    def start_collect(self, kind, seconds, **kw):
        """Avvia una raccolta dati temporizzata (rumore, calibrazione, addestramento)."""
        if self.source is None:
            self.status('Avviare prima l\'acquisizione (scheda Live) con la sorgente desiderata.', err=True)
            return False
        self.collect = dict(kind=kind, until=time.time() + seconds, raw=[], seconds=seconds, n=0, **kw)
        self.status(f'Raccolta "{kind}" in corso per {seconds:.0f} s ...')
        return True

    # ================================================================== CICLO PRINCIPALE
    def tick(self):
        src = self.source
        if src is None:
            return
        if src.error:
            self.status(f'Errore sorgente: {src.error}', err=True)
        data = src.read_all()
        if data is None:
            return
        if self.recorder is not None:
            self.recorder.write(data)
        if self.collect is not None and self.collect['kind'] in ('noise', 'check', 'gain', 'phase', 'orient', 'spl'):
            self.collect['raw'].append(data)
        nwin = int(self.cfg.window_s * src.fs)
        self.buf = np.concatenate([self.buf, data], 0)[-nwin:]
        if len(self.buf) < min(nwin, self.cfg.block_size * 2):
            return
        t = time.perf_counter()
        try:
            r = self.proc.process(self.buf)
        except Exception as exc:
            self.status(f'Errore elaborazione: {exc}', err=True)
            traceback.print_exc()
            return
        if r is None:
            return
        self.proc_ms = 0.8 * self.proc_ms + 0.2 * (time.perf_counter() - t) * 1000
        # --- riconoscimento impronta
        cls, pdrone, probs = (None, np.nan, {})
        if self.model.trained:
            try:
                cls, pdrone, probs = self.model.predict(r['focused'], r['freqs'])
            except Exception:
                pass
        has_noise = self.noise.has_profile(len(r['freqs']))
        detected = r['dir_index_db'] >= self.cfg.detect_threshold_db
        if has_noise and np.isfinite(r['snr_band_db']):
            detected = detected and r['snr_band_db'] >= self.cfg.snr_detect_db
        if self.cfg.use_classifier and self.model.trained:
            detected = detected and (pdrone >= self.cfg.classifier_threshold)
        # --- rumore adattivo (solo quando non c'e' rilevamento)
        if self.cfg.adaptive_noise and not detected and has_noise and self.collect is None \
                and r['snr_band_db'] < min(3.0, self.cfg.snr_detect_db):
            self.noise.adapt(r['C_all'], self.cfg.adaptive_alpha)
        # --- tracking
        now = time.time()
        extra = dict(dir_db=round(float(r['dir_index_db']), 1), snr_db=None if not np.isfinite(r['snr_band_db'])
                     else round(float(r['snr_band_db']), 1), az=round(float(r['az']), 2), el=round(float(r['el']), 2), level_db=round(float(r['level_db']), 1),
                     contrast_db=round(float(r['contrast_db']), 1), p_drone=None if not np.isfinite(pdrone)
                     else round(float(pdrone), 3), classe=cls)
        est = self.tracker.update(r['peak'], detected, now, extra)
        self.last = dict(r=r, detected=detected, cls=cls, pdrone=pdrone, probs=probs, est=est)
        # --- raccolte in corso
        self._collect_step(r)
        # --- grafica
        self._update_views(r, detected, cls, pdrone, probs, est)
        self._n_ticks = getattr(self, '_n_ticks', 0) + 1
        if self._n_ticks % 8 == 0:
            self._plot_noise()

    def _collect_step(self, r):
        c = self.collect
        if c is None:
            return
        if c['kind'] == 'train':
            self.model.add_sample(c['label'], r['focused'], r['freqs'])
            c['n'] += 1
        remaining = c['until'] - time.time()
        if remaining > 0:
            self.status(f'Raccolta "{c["kind"]}" in corso: {remaining:.0f} s rimanenti ...')
            return
        self.collect = None
        try:
            self._finish_collect(c)
        except Exception as exc:
            self.status(f'Errore durante "{c["kind"]}": {exc}', err=True)
            traceback.print_exc()

    def _finish_collect(self, c):
        kind = c['kind']
        if kind == 'train':
            self._refresh_training()
            self.status(f'Aggiunti {c["n"]} campioni alla classe "{c["label"]}". Premere "Addestra".')
            return
        raw = np.concatenate(c['raw'], 0) if c['raw'] else None
        if raw is None or len(raw) < self.cfg.block_size * 4:
            self.status('Dati insufficienti.', err=True)
            return
        pr = self.proc
        if kind == 'noise':
            self.noise.start_recording()
            n = int(self.cfg.window_s * pr.fs)
            for s in range(0, len(raw) - n + 1, n):
                _, C = pr.spectra(raw[s:s + n], all_channels=True)
                self.noise.add_frame(C, pr.freqs, n / pr.fs)
            rep = self.noise.finish_recording(pr.pos_all, pr.c)
            self.noise.save(noise_file())
            self.noise_report.text = rep
            self.noise_suggest = self.noise.suggestion
            if c.get('auto'):
                self._apply_preset(self.noise.suggestion['preset'])
                self.cfg.f_min = self.noise.suggestion['f_min']
                self.band_slider.value = (self.cfg.f_min, self.cfg.f_max)
                self.cfg.csm_subtraction = True if self.cfg.environment != 'Campo aperto' else self.cfg.csm_subtraction
                self._sync_noise_widgets()
                self.rebuild()
                self.status(f'Autocalibrazione ambiente completata: preset "{self.cfg.environment}", '
                            f'banda {self.cfg.f_min:.0f}-{self.cfg.f_max:.0f} Hz.')
            else:
                self.status(f'Profilo rumore registrato ({self.noise.duration:.0f} s) e salvato.')
            self._plot_noise()
            return
        # calibrazioni: CSM grezza (senza correzioni) su tutti i canali
        _, C = pr.spectra(raw, all_channels=True, calibrated=(kind in ('orient', 'spl')))
        spos = np.array([self.cal_x.value, self.cal_y.value, self.cal_z.value], float)
        if kind == 'check':
            rep, invalid, rel = check_channels(C, pr.freqs)
            self.cal_report.text = rep
            self.ch_src.data = dict(ch=[str(i) for i in range(len(rel))], lvl=list(rel),
                                    color=['#b00020' if i in invalid else '#3288bd' for i in range(len(rel))])
            self.suggested_invalid = invalid
            self.status('Verifica canali completata. Usare "Applica canali suggeriti" per escluderli.')
        elif kind == 'gain':
            g, rep = gain_calibration(C, pr.freqs, pr.pos_all, spos, self.cfg.f_min, self.cfg.f_max,
                                      self.cfg.invalid_channels)
            self.calib.gains = g
            self.calib.report = rep
            self.calib.save(CALIB_FILE)
            self.cal_report.text = rep
            self.ch_src.data = dict(ch=[str(i) for i in range(len(g))], lvl=list(20 * np.log10(g)),
                                    color=['#3288bd'] * len(g))
            self.rebuild()
            self.status('Calibrazione guadagni completata e salvata.')
        elif kind == 'phase':
            ph, rep = phase_calibration(C, pr.freqs, pr.pos_all, spos, pr.c)
            self.calib.phase = ph
            self.calib.phase_block = self.cfg.block_size
            self.calib.save(CALIB_FILE)
            self.cal_report.text = rep
            self.status('Calibrazione di fase completata. Attivarla con la casella "usa correzione fase".')
        elif kind == 'orient':
            raw_pos = load_positions(self.cfg.geometry)
            exp_xy = pr.grid.from_position(spos)
            best, rep = find_orientation(C, pr.freqs, raw_pos, pr.grid, exp_xy, pr.c, self.cfg.far_field,
                                         max(self.cfg.f_min, 800), self.cfg.f_max,
                                         active=self.proc.active)
            self.cfg.flip_x, self.cfg.flip_y, self.cfg.swap_xy = best
            self.orient_cb.active = [i for i, v in enumerate(best) if v]
            self.cal_report.text = rep + f'\n=> Orientamento scelto: flip_x={best[0]} flip_y={best[1]} swap_xy={best[2]}'
            self.cfg.to_json()
            self.rebuild()
            self.status('Orientamento dell\'array determinato e applicato.')
        elif kind == 'spl':
            if self.last is None:
                return
            lvl = self.last['r']['level_db'] - self.cfg.spl_offset_db
            self.cfg.spl_offset_db = float(self.spl_ref.value) - lvl
            self.cfg.to_json()
            self.cal_report.text = f'Offset livello assoluto: {self.cfg.spl_offset_db:+.1f} dB ' \
                                   f'(livello misurato {lvl:.1f} dB rel. -> {self.spl_ref.value:.1f} dB SPL)'
            self.status('Calibrazione livello assoluto salvata.')

    # ================================================================== AGGIORNAMENTO GRAFICI
    def _update_views(self, r, detected, cls, pdrone, probs, est):
        g = self.proc.grid
        m = np.nan_to_num(r['map_db'], nan=-200.0)
        s = g.step
        x0, x1, y0, y1 = g.extent
        self.map_src.data = dict(image=[m], x=[x0 - s / 2], y=[y0 - s / 2], dw=[x1 - x0 + s], dh=[y1 - y0 + s])
        self.peak_src.data = dict(x=[r['peak'][0]], y=[r['peak'][1]])
        if est is not None:
            self.trk_src.data = dict(x=[est[0]], y=[est[1]], c=['red' if detected else 'orange'])
            hist = self.tracker.history[-self.cfg.trail_length:]
            self.trail_src.data = dict(x=[h['x'] for h in hist], y=[h['y'] for h in hist])
        else:
            self.trk_src.data = dict(x=[], y=[], c=[])
        if isinstance(self.source, audio_io.SimulatedSource):
            tx, ty = g.from_position(self.source.true_pos)
            self.true_src.data = dict(x=[tx], y=[ty])
        # info
        det = '<span style="background:#c62828;color:white;padding:3px 8px;border-radius:4px">SORGENTE RILEVATA</span>' \
            if detected else '<span style="background:#9e9e9e;color:white;padding:3px 8px;border-radius:4px">nessun rilevamento</span>'
        cl = ''
        if cls is not None:
            cl = f'<br>Classe: <b>{cls}</b> &nbsp; P(drone) = <b>{pdrone * 100:.0f} %</b>'
            self.prob_src.data = dict(cls=list(probs.keys()), p=[float(v) for v in probs.values()])
        snr = '' if not np.isfinite(r['snr_band_db']) else f'<br>SNR banda vs fondo: {r["snr_band_db"]:.1f} dB'
        self.info_div.text = (f'{det}<br><br>Azimut: <b>{r["az"]:.1f}°</b> &nbsp; Elevazione: <b>{r["el"]:.1f}°</b>'
                              f'<br>Posizione griglia: ({r["peak"][0]:+.3f}, {r["peak"][1]:+.3f})'
                              f'<br>Direzionalità (λ1/λ medio): {r["dir_index_db"]:.1f} dB'
                              f'<br>Contrasto mappa: {r["contrast_db"]:.1f} dB'
                              f'<br>Livello banda (focalizzato): {r["level_db"]:.1f} dB{snr}{cl}'
                              f'<br><span style="color:#888">elaborazione {self.proc_ms:.0f} ms, overflow {self.source.overflows if self.source else 0}</span>')
        # spettro
        f = r['freqs'][1:]
        mode = self.spec_mode.value
        off = self.cfg.spl_offset_db
        psd = r['psd'][1:]
        if mode.startswith('Tutti'):
            xs = [f] * psd.shape[1]
            ys = [db(psd[:, i]) + off for i in range(psd.shape[1])]
            cols = [CH_COLORS[self.proc.active[i] % 16] for i in range(psd.shape[1])]
        elif mode.startswith('Canale'):
            ch = int(mode.split()[1])
            if ch in list(self.proc.active):
                i = list(self.proc.active).index(ch)
                xs, ys, cols = [f], [db(psd[:, i]) + off], [CH_COLORS[ch]]
            else:
                xs, ys, cols = [], [], []
        elif mode.startswith('Focal'):
            xs, ys, cols = [f], [db(r['focused'][1:]) + off], ['#c62828']
        else:
            xs, ys, cols = [f], [db(psd.mean(1)) + off], ['#3288bd']
        self.spec_lines.data = dict(xs=xs, ys=ys, color=cols, label=[''] * len(xs))
        if self.noise.has_profile(len(r['freqs'])):
            self.spec_noise.data = dict(x=f, y=db(self.noise.psd[1:, self.proc.active].mean(1)) + off)
        # spettrogramma
        self.spec_img = np.roll(self.spec_img, -1, axis=1)
        self.spec_img[:, -1] = db(r['focused']) + off
        self.sg_src.data = dict(image=[self.spec_img], x=[0], y=[0], dw=[SPEC_FRAMES], dh=[self.proc.fs / 2])
        # traiettoria
        if est is not None and self.tracker.history:
            h = self.tracker.history[-1]
            self.traj_src.stream(dict(x=[h['x']], y=[h['y']], t=[h['t'] - self.t0], az=[h['az']], el=[h['el']],
                                      lvl=[h['level_db']]), rollover=5000)

    # ================================================================== TAB TRAIETTORIA
    def _build_trajectory(self):
        self.traj_src = ColumnDataSource(data=dict(x=[], y=[], t=[], az=[], el=[], lvl=[]))
        tmap = LinearColorMapper(palette=Turbo256, low=0, high=60)
        self.traj_map = tmap
        self.traj_fig = figure(title='Traiettoria stimata (colore = tempo)', width=580, height=520,
                               x_range=Range1d(-1, 1), y_range=Range1d(-1, 1), tools='pan,wheel_zoom,reset,save,hover',
                               tooltips=[('t [s]', '@t{0.0}'), ('az', '@az{0.0}°'), ('el', '@el{0.0}°'),
                                         ('livello', '@lvl{0.0} dB')])
        self.traj_fig.toolbar.logo = None
        self.traj_horizon = ColumnDataSource(data=dict(xs=[], ys=[]))
        self.traj_fig.multi_line('xs', 'ys', source=self.traj_horizon, line_color='#9e9e9e', line_dash='dotted')
        self.traj_fig.line('x', 'y', source=self.traj_src, line_color='#555', line_alpha=0.4)
        self.traj_fig.scatter('x', 'y', source=self.traj_src, size=6,
                              color={'field': 't', 'transform': tmap})
        self.traj_fig.add_layout(ColorBar(color_mapper=tmap, title='t / s'), 'right')
        self.az_fig = figure(title='Azimut', width=560, height=250, x_axis_label='t / s', y_axis_label='gradi',
                             tools='pan,wheel_zoom,reset,save')
        self.az_fig.scatter('t', 'az', source=self.traj_src, size=4, color='#3288bd')
        self.el_fig = figure(title='Elevazione sopra il piano dell\'array', width=560, height=250,
                             x_axis_label='t / s', y_axis_label='gradi', x_range=self.az_fig.x_range,
                             tools='pan,wheel_zoom,reset,save')
        self.el_fig.scatter('t', 'el', source=self.traj_src, size=4, color='#c62828')
        for f in (self.az_fig, self.el_fig):
            f.toolbar.logo = None
        self.traj_src.on_change('data', self._traj_color_range)
        exp_btn = Button(label='Esporta CSV', button_type='primary', width=140)
        exp_btn.on_click(self._export_traj)
        clr = Button(label='Azzera', width=100)
        clr.on_click(self._clear_track)
        self.traj_div = Div(text='')
        self.gate_sp = sp('Gate associazione', self.cfg.track_gate, 0.05, 0.02, 5)
        self.q_sp = sp('Rumore di processo', self.cfg.track_process_noise, 0.05, 0.01, 10)
        self.miss_sp = sp('Max frame persi', self.cfg.track_max_missed, 1, 1, 100)
        for w, a in ((self.gate_sp, 'track_gate'), (self.q_sp, 'track_process_noise'),
                     (self.miss_sp, 'track_max_missed')):
            w.on_change('value', lambda _a, _o, n, a=a: self._set_tracker(a, n))
        note = Div(text='<i>Il filtro di Kalman (velocità costante) smussa le misure e scarta i salti '
                        'fuori dal gate (riflessioni, sorgenti secondarie). Nella mappa (u,v) i cerchi '
                        'tratteggiati indicano elevazione 0°, 30°, 60°.</i>', width=520)
        self.traj_layout = column(row(exp_btn, clr, self.gate_sp, self.q_sp, self.miss_sp), self.traj_div,
                                  row(self.traj_fig, column(self.az_fig, self.el_fig)), note)

    def _traj_color_range(self, _a, _o, new):
        if new['t']:
            self.traj_map.low, self.traj_map.high = float(new['t'][0]), float(new['t'][-1]) + 1e-3

    def _set_tracker(self, attr, value):
        setattr(self.cfg, attr, value)
        self.tracker.gate = self.cfg.track_gate
        self.tracker.q = self.cfg.track_process_noise
        self.tracker.max_missed = int(self.cfg.track_max_missed)

    def _export_traj(self):
        path = DATA_DIR / f'traiettoria_{datetime.now():%Y%m%d_%H%M%S}.csv'
        n = self.tracker.export_csv(path)
        self.traj_div.text = f'Esportati {n} punti in <code>{path}</code>' if n else 'Nessun punto da esportare.'

    # ================================================================== TAB CALIBRAZIONE
    def _build_calibration(self):
        self.suggested_invalid = []
        self.cal_dur = Slider(title='Durata acquisizione (s)', start=2, end=30, step=1, value=5, width=250)
        self.cal_x = sp('Sorgente x (m)', 0.0, 0.1)
        self.cal_y = sp('Sorgente y (m)', 0.0, 0.1)
        self.cal_z = sp('Sorgente z (m)', 2.0, 0.1, 0.2)
        self.cal_x.on_change('value', self._show_expected)
        self.cal_y.on_change('value', self._show_expected)
        self.cal_z.on_change('value', self._show_expected)
        self.spl_ref = sp('SPL fonometro (dB)', 80.0, 0.5, 0, 140, width=140)
        btns = []
        for label, kind in (('1. Verifica canali', 'check'), ('2. Calibra guadagni', 'gain'),
                            ('2b. Calibra fase (avanzato)', 'phase'), ('3. Determina orientamento', 'orient'),
                            ('4. Livello assoluto (SPL)', 'spl')):
            b = Button(label=label, width=220, button_type='primary')
            b.on_click(lambda k=kind: self.start_collect(k, self.cal_dur.value))
            btns.append(b)
        apply_inv = Button(label='Applica canali suggeriti', width=220)
        apply_inv.on_click(self._apply_suggested_invalid)
        reset = Button(label='Reset calibrazione', width=160, button_type='danger')
        reset.on_click(self._reset_calib)
        self.calib_use = CheckboxGroup(labels=['usa guadagni', 'usa correzione fase'],
                                       active=[i for i, v in enumerate([self.cfg.use_gain_calibration,
                                                                         self.cfg.use_phase_calibration]) if v])
        self.calib_use.on_change('active', self._on_calib_use)
        self.orient_cb = CheckboxGroup(labels=['flip x', 'flip y', 'scambia x/y'],
                                       active=[i for i, v in enumerate([self.cfg.flip_x, self.cfg.flip_y,
                                                                         self.cfg.swap_xy]) if v], inline=True)
        self.orient_cb.on_change('active', self._on_orient)
        self.cal_report = PreText(text=self.calib.report or 'Nessuna calibrazione eseguita.', width=560, height=330)
        self.ch_src = ColumnDataSource(data=dict(ch=[str(i) for i in range(16)], lvl=[0.0] * 16,
                                                 color=['#3288bd'] * 16))
        chf = figure(title='Livelli / guadagni per canale (dB)', x_range=[str(i) for i in range(16)], width=520,
                     height=260, tools='save')
        chf.vbar(x='ch', top='lvl', width=0.8, color='color', source=self.ch_src)
        chf.toolbar.logo = None
        self.geo_src = ColumnDataSource(data=dict(x=[], y=[], ch=[], color=[]))
        gf = figure(title='Geometria microfoni (coordinate array, dopo orientamento)', width=380, height=380,
                    match_aspect=True, tools='save', x_range=(-0.1, 0.1), y_range=(-0.1, 0.1),
                    x_axis_label='x / m', y_axis_label='y / m')
        gf.toolbar.logo = None
        gf.scatter('x', 'y', source=self.geo_src, size=26, color='color', fill_alpha=0.25)
        gf.add_layout(LabelSet(x='x', y='y', text='ch', source=self.geo_src, text_align='center',
                               text_baseline='middle', text_font_size='10pt'))
        help_div = Div(width=560, text='''
<b>Procedura</b> (avviare prima l'acquisizione nella scheda <i>Live</i>):<br>
<b>1.</b> Verifica canali: pochi secondi con una sorgente a larga banda accesa (rumore rosa).<br>
<b>2.</b> Guadagni: altoparlante <u>fisso</u> in posizione nota (consigliato sull'asse: x=0, y=0, z≥1.5 m),
rumore rosa ≥ 20 dB sopra il fondo. La fase (2b) corregge anche ritardi/fasi per canale.<br>
<b>3.</b> Orientamento: spostare l'altoparlante fuori asse (es. x=+1 m, y=0, z=2 m) e inserire la posizione:
il programma prova le 8 combinazioni di ribaltamento degli assi e sceglie quella corretta.<br>
<b>4.</b> SPL assoluto: fonometro accanto all'array, inserire il valore letto (pesatura Z, stessa banda).<br>
Coordinate: origine al centro dell'array, z lungo la normale (verso la sorgente/il cielo).''')
        self.cal_layout = column(
            row(column(help_div, row(self.cal_x, self.cal_y, self.cal_z), self.cal_dur,
                       row(btns[0], apply_inv), row(btns[1], btns[2]), row(btns[3]), row(btns[4], self.spl_ref),
                       Div(text='<b>Opzioni</b>'), self.calib_use, self.orient_cb, reset),
                column(self.cal_report, chf), gf))

    def _show_expected(self, _a, _o, _n):
        try:
            gx, gy = self.proc.grid.from_position([self.cal_x.value, self.cal_y.value, self.cal_z.value])
            self.true_src.data = dict(x=[gx], y=[gy])
        except Exception:
            pass

    def _apply_suggested_invalid(self):
        self.cfg.invalid_channels = list(self.suggested_invalid)
        self.invalid_mc.value = [str(i) for i in self.suggested_invalid]
        self.rebuild()
        self.status(f'Canali esclusi: {self.cfg.invalid_channels}')

    def _reset_calib(self):
        self.calib.reset()
        if CALIB_FILE.exists():
            CALIB_FILE.unlink()
        self.cal_report.text = 'Calibrazione azzerata.'
        self.rebuild()

    def _on_calib_use(self, _a, _o, new):
        self.cfg.use_gain_calibration = 0 in new
        self.cfg.use_phase_calibration = 1 in new
        self.rebuild()

    def _on_orient(self, _a, _o, new):
        self.cfg.flip_x, self.cfg.flip_y, self.cfg.swap_xy = (0 in new), (1 in new), (2 in new)
        self.rebuild()

    # ================================================================== TAB RUMORE / AMBIENTE
    def _build_noise(self):
        cfg = self.cfg
        self.noise_suggest = {}
        self.env_select = Select(title='Ambiente (preset)', options=list(ENVIRONMENT_PRESETS), value=cfg.environment,
                                 width=220)
        self.env_select.on_change('value', lambda a, o, n: self._apply_preset(n, sync=True))
        self.env_note = Div(text=ENVIRONMENT_PRESETS.get(cfg.environment, {}).get('note', ''), width=520)
        self.noise_dur = Slider(title='Durata registrazione fondo (s)', start=3, end=60, step=1, value=10, width=250)
        rec = Button(label='Registra rumore di fondo', button_type='primary', width=220)
        rec.on_click(lambda: self.start_collect('noise', self.noise_dur.value))
        auto = Button(label='Autocalibrazione ambiente', button_type='success', width=220)
        auto.on_click(lambda: self.start_collect('noise', self.noise_dur.value, auto=True))
        clr = Button(label='Cancella profilo', width=140, button_type='danger')
        clr.on_click(self._clear_noise)
        self.noise_report = PreText(text=self.noise.report or 'Nessun profilo di rumore.', width=560, height=160)
        self.nw = {}
        self.nw['diag_removal'] = checkbox('Rimozione diagonale CSM (rumore incoerente: vento, elettronica)',
                                           cfg.diag_removal)
        self.nw['csm_subtraction'] = checkbox('Sottrazione CSM del rumore di fondo (rumore coerente)',
                                              cfg.csm_subtraction)
        self.nw['adaptive_noise'] = checkbox('Aggiornamento adattivo del fondo (quando non c\'e\' rilevamento)',
                                             cfg.adaptive_noise)
        self.nw['stability_weighting'] = checkbox('Pesatura stabilità tonale (anti-uccelli/transitori)',
                                                  cfg.stability_weighting)
        self.nw['signature_weighting'] = checkbox('Pesatura frequenze con impronta addestrata',
                                                  cfg.signature_weighting)
        self.nw['freq_norm'] = checkbox('Normalizza ogni frequenza prima della somma a banda larga', cfg.freq_norm)
        for k, w in self.nw.items():
            w.on_change('active', lambda _a, _o, n, k=k: self._set(k, 0 in n, rebuild=False))
        self.ns = {}
        self.ns['csm_sub_factor'] = Slider(title='Fattore sottrazione CSM', start=0, end=2, step=0.05,
                                           value=cfg.csm_sub_factor, width=250)
        self.ns['adaptive_alpha'] = Slider(title='Velocità adattamento fondo', start=0.005, end=0.3, step=0.005,
                                           value=cfg.adaptive_alpha, width=250)
        self.ns['csm_smoothing'] = Slider(title='Media temporale CSM', start=0, end=0.95, step=0.05,
                                          value=cfg.csm_smoothing, width=250)
        self.ns['snr_mask_db'] = Slider(title='Maschera SNR per frequenza (dB)', start=-10, end=20, step=0.5,
                                        value=cfg.snr_mask_db, width=250)
        for k, w in self.ns.items():
            w.on_change('value', lambda _a, _o, n, k=k: self._set(k, float(n), rebuild=False))
        self.noise_fig = figure(title='Profilo rumore di fondo vs segnale attuale', width=620, height=320,
                                x_axis_type='log', x_axis_label='f / Hz', y_axis_label='dB (rel.)',
                                tools='pan,wheel_zoom,reset,save')
        self.noise_fig.toolbar.logo = None
        self.noise_fig.x_range = Range1d(50, cfg.sample_rate / 2)
        self.np_src = ColumnDataSource(data=dict(x=[], y=[]))
        self.ns_src = ColumnDataSource(data=dict(x=[], y=[]))
        self.noise_fig.line('x', 'y', source=self.np_src, color='gray', line_width=2, legend_label='fondo')
        self.noise_fig.line('x', 'y', source=self.ns_src, color='#3288bd', legend_label='attuale (media canali)')
        upd = Button(label='Aggiorna grafico', width=140)
        upd.on_click(self._plot_noise)
        help_div = Div(width=560, text='''
<b>Come si rimuove il rumore di fondo</b><br>
• <b>Vento / campo aperto</b>: il vento genera pressione turbolenta <i>diversa su ogni microfono</i> (incoerente):
si concentra sulla diagonale della matrice di cross-spettro (CSM) e si elimina rimuovendola; si alza inoltre la
frequenza minima (il vento è forte sotto 300–500 Hz). Usare una cuffia antivento.<br>
• <b>Foresta</b>: fruscio diffuso + cinguettii (transitori modulati in 2–8 kHz). Sottrazione della CSM di fondo,
pesatura della stabilità tonale (le armoniche dei rotori sono stabili nel tempo) e classificatore addestrato
con la classe “uccelli”.<br>
• <b>Canyon urbano</b>: traffico coerente a bassa frequenza e <i>riflessioni</i> sulle facciate (sorgenti immagine).
Sottrazione CSM adattiva, f<sub>min</sub> ≥ 1 kHz, metodi ad alta risoluzione (MUSIC/Capon), gate stretto nel tracker.<br>
• <b>Autocalibrazione</b>: registra il fondo <u>senza</u> il drone, analizza livello, coerenza tra microfoni e
variabilità, sceglie il preset e la banda di analisi. Il fondo poi si aggiorna da solo se “adattivo” è attivo.''')
        self.noise_layout = column(row(
            column(self.env_select, self.env_note, self.noise_dur, row(rec, auto), clr,
                   *self.nw.values(), *self.ns.values()),
            column(self.noise_report, upd, self.noise_fig, help_div)))

    def _apply_preset(self, name, sync=False):
        p = ENVIRONMENT_PRESETS.get(name, {})
        self.cfg.environment = name
        for k, v in p.items():
            if k != 'note' and hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
        self.env_note.text = p.get('note', '')
        if self.env_select.value != name:
            self.env_select.value = name
        if sync:
            self._sync_noise_widgets()
            self.rebuild()

    def _sync_noise_widgets(self):
        cfg = self.cfg
        for k, w in self.nw.items():
            w.active = [0] if getattr(cfg, k) else []
        for k, w in self.ns.items():
            w.value = getattr(cfg, k)
        self.band_slider.value = (cfg.f_min, cfg.f_max)
        self.band_box.left, self.band_box.right = cfg.f_min, cfg.f_max
        self.method_select.value = cfg.method
        self.thr_slider.value = cfg.detect_threshold_db

    def _clear_noise(self):
        self.noise.clear()
        if noise_file().exists():
            noise_file().unlink()
        self.noise_report.text = 'Profilo cancellato.'
        self.np_src.data = dict(x=[], y=[])
        self.spec_noise.data = dict(x=[], y=[])

    def _plot_noise(self):
        if self.noise.has_profile():
            f = self.noise.freqs[1:]
            self.np_src.data = dict(x=f, y=db(self.noise.psd[1:].mean(1)))
        if self.last is not None:
            r = self.last['r']
            self.ns_src.data = dict(x=r['freqs'][1:], y=db(r['psd'][1:].mean(1)))

    # ================================================================== TAB ADDESTRAMENTO
    def _build_training(self):
        cfg = self.cfg
        self.cls_select = Select(title='Classe', options=CLASS_PRESETS, value='drone', width=150)
        self.cls_text = TextInput(title='...oppure nome nuova classe', value='', width=200)
        self.train_dur = Slider(title='Durata raccolta (s)', start=3, end=120, step=1, value=15, width=250)
        rec = Button(label='Registra campioni dal vivo', button_type='primary', width=220)
        rec.on_click(lambda: self.start_collect('train', self.train_dur.value, label=self._cls_name()))
        self.wav_input = FileInput(accept='.wav,.flac,.ogg', multiple=True, width=300)
        self.wav_input.on_change('value', self._on_wav)
        train = Button(label='Addestra modello', button_type='success', width=180)
        train.on_click(self._train)
        delc = Button(label='Elimina classe selezionata', width=200, button_type='danger')
        delc.on_click(self._del_class)
        self.use_clf = checkbox('Usa il classificatore per confermare il rilevamento', cfg.use_classifier)
        self.use_clf.on_change('active', lambda a, o, n: self._set('use_classifier', 0 in n, rebuild=False))
        self.clf_thr = Slider(title='Soglia P(drone)', start=0.05, end=0.99, step=0.01, value=cfg.classifier_threshold,
                              width=250)
        self.clf_thr.on_change('value', lambda a, o, n: self._set('classifier_threshold', n, rebuild=False))
        self.train_report = PreText(text=self.model.report, width=560, height=80)
        self.counts_div = Div(text='', width=500)
        self.prob_src = ColumnDataSource(data=dict(cls=[], p=[]))
        self.prob_fig = figure(title='Probabilità per classe (live)', y_range=CLASS_PRESETS, x_range=(0, 1),
                               width=420, height=260, tools='')
        self.prob_fig.hbar(y='cls', right='p', height=0.7, source=self.prob_src, color='#c62828')
        self.prob_fig.toolbar.logo = None
        self.cls_fig = figure(title='Spettro medio per classe (impronta)', width=620, height=320, x_axis_type='log',
                              x_axis_label='f / Hz', y_axis_label='dB (rel.)', tools='pan,wheel_zoom,reset,save')
        self.cls_fig.toolbar.logo = None
        self.cls_src = ColumnDataSource(data=dict(xs=[], ys=[], color=[], name=[]))
        self.cls_fig.multi_line('xs', 'ys', color='color', source=self.cls_src, line_width=2)
        self.cls_fig.add_tools(HoverTool(tooltips=[('classe', '@name')]))
        help_div = Div(width=560, text='''
<b>Addestramento dell'impronta acustica</b><br>
1. Avviare l'acquisizione. Riprodurre con l'altoparlante (fisso e poi in movimento) i suoni di drone e
registrare campioni nella classe <b>drone</b> (i campioni sono presi dal segnale <i>focalizzato</i> sulla
sorgente, quindi con meno rumore). Si possono creare classi specifiche, es. <i>drone_dji</i>, <i>drone_fpv</i>:
tutte le classi che contengono "drone" contano come positive.<br>
2. Registrare le classi negative: <b>sfondo</b> (senza sorgente), <b>uccelli</b>, <b>traffico</b>, <b>voci</b>...<br>
3. In alternativa/aggiunta caricare file WAV (mono o multicanale) nella classe selezionata.<br>
4. Premere <b>Addestra</b>: viene riportata l'accuratezza in validazione incrociata. Attivare
"Usa il classificatore" e, nella scheda Rumore, "Pesatura frequenze con impronta" per focalizzare la
mappa sulle bande tipiche del drone.''')
        self.train_layout = column(row(
            column(help_div, row(self.cls_select, self.cls_text), self.train_dur, rec,
                   Div(text='<b>Carica file audio nella classe selezionata:</b>'), self.wav_input,
                   row(train, delc), self.use_clf, self.clf_thr, self.counts_div),
            column(self.train_report, self.prob_fig, self.cls_fig)))
        self._refresh_training()

    def _cls_name(self):
        return (self.cls_text.value.strip() or self.cls_select.value).replace(' ', '_')

    def _on_wav(self, _a, _o, new):
        import soundfile as sf
        vals = new if isinstance(new, list) else [new]
        names = self.wav_input.filename if isinstance(self.wav_input.filename, list) else [self.wav_input.filename]
        tot = 0
        for v, name in zip(vals, names):
            try:
                data, fs = sf.read(io.BytesIO(base64.b64decode(v)), dtype='float32')
                tot += self.model.add_wav(self._cls_name(), data, fs, self.cfg.sample_rate, self.cfg.block_size,
                                          self.cfg.window_s)
            except Exception as exc:
                self.status(f'Errore lettura {name}: {exc}', err=True)
        self._refresh_training()
        self.status(f'Aggiunti {tot} campioni da {len(vals)} file alla classe "{self._cls_name()}".')

    def _train(self):
        rep = self.model.train()
        self.train_report.text = rep
        if self.model.trained:
            self.model.save(model_file())
            self._update_signature_weights()
        self._refresh_training()

    def _del_class(self):
        self.model.remove_class(self._cls_name())
        self._refresh_training()

    def _refresh_training(self):
        cnt = self.model.counts()
        self.counts_div.text = '<b>Campioni raccolti:</b> ' + (', '.join(f'{k}: {v}' for k, v in cnt.items()) or '—')
        classes = list(dict.fromkeys(CLASS_PRESETS + list(cnt)))
        self.prob_fig.y_range.factors = classes
        opts = list(dict.fromkeys(CLASS_PRESETS + list(cnt)))
        self.cls_select.options = opts
        xs, ys, cols, names = [], [], [], []
        pal = list(Category20[20])
        for i, (k, v) in enumerate(self.model.class_psd.items()):
            if v and self.model.freqs is not None:
                xs.append(self.model.freqs[1:])
                ys.append(np.mean(v, 0)[1:])
                cols.append(pal[(2 * i) % 20])
                names.append(k)
        self.cls_src.data = dict(xs=xs, ys=ys, color=cols, name=names)

    # ================================================================== TAB PARAMETRI
    def _build_params(self):
        cfg = self.cfg
        devs = audio_io.list_input_devices(1)
        self.dev_select = Select(title='Dispositivo di ingresso (vuoto = ricerca automatica UMA-16)',
                                 options=[''] + devs, value=cfg.device_name if cfg.device_name in devs else '',
                                 width=460)
        ref = Button(label='Aggiorna elenco dispositivi', width=200)
        ref.on_click(lambda: setattr(self.dev_select, 'options', [''] + audio_io.list_input_devices(1)))
        self.p = {}
        self.p['sample_rate'] = Select(title='Frequenza di campionamento', value=str(cfg.sample_rate),
                                       options=['48000', '44100', '32000', '16000', '11025'], width=150)
        self.rec_select = Select(title='Registrazioni salvate', options=[], width=360)
        self.rec_select.on_change('value', lambda a, o, n: setattr(self.p['replay_file'], 'value', n) if n else None)
        self.p['replay_file'] = TextInput(title='File da riprodurre (.h5 acoular / .wav 16 canali)',
                                          value=cfg.replay_file, width=460)
        self.p['geometry'] = Select(title='Geometria microfoni', options=available_geometries(), value=cfg.geometry,
                                    width=260)
        self.invalid_mc = MultiChoice(title='Canali esclusi', options=[str(i) for i in range(16)],
                                      value=[str(i) for i in cfg.invalid_channels], width=460)
        self.p['temperature_c'] = sp('Temperatura (°C)', cfg.temperature_c, 0.5, -30, 50)
        self.p['humidity'] = sp('Umidità (%)', cfg.humidity, 5, 0, 100)
        self.c_div = Div(text=f'c = {cfg.c:.1f} m/s')
        self.p['block_size'] = Select(title='Blocco FFT', value=str(cfg.block_size),
                                      options=['256', '512', '1024', '2048', '4096'], width=110)
        self.p['window_s'] = sp('Finestra analisi (s)', cfg.window_s, 0.05, 0.1, 5)
        self.p['update_ms'] = sp('Aggiornamento (ms)', cfg.update_ms, 50, 100, 5000)
        self.p['grid_mode'] = Select(title='Griglia', options=['Emisfero (u,v)', 'Piano (x,y @ z)'],
                                     value=cfg.grid_mode, width=170)
        self.p['grid_res'] = sp('Passo griglia', cfg.grid_res, 0.005, 0.005, 1)
        self.p['max_off_axis_deg'] = sp('Max angolo da asse (°)', cfg.max_off_axis_deg, 5, 10, 90)
        for k in ('x_min', 'x_max', 'y_min', 'y_max'):
            self.p[k] = sp(f'{k} (m)', getattr(cfg, k), 0.1)
        self.p['z'] = sp('Distanza z (m)', cfg.z, 0.1, 0.1)
        self.p['far_field'] = checkbox('Campo lontano (onde piane)', cfg.far_field)
        self.p['functional_gamma'] = sp('γ funzionale', cfg.functional_gamma, 1, 1, 100)
        self.p['music_sources'] = sp('N. sorgenti (MUSIC)', cfg.music_sources, 1, 1, 8)
        self.p['capon_loading'] = sp('Carico diagonale Capon', cfg.capon_loading, 0.01, 0.0, 1)
        self.p['sim_bpf_hz'] = sp('Sim.: BPF (Hz)', cfg.sim_bpf_hz, 10, 30, 1000)
        self.p['sim_snr_db'] = sp('Sim.: SNR (dB)', cfg.sim_snr_db, 1, -20, 40)
        self.p['sim_path'] = Select(title='Sim.: traiettoria', options=['Cerchio', 'Linea', 'Fisso'],
                                    value=cfg.sim_path, width=120)
        self.sim_on = checkbox('Sim.: drone acceso (spegnere per registrare il fondo)', True)
        self.sim_on.on_change('active', lambda a, o, n: setattr(self.source, 'drone_on', 0 in n)
                              if isinstance(self.source, audio_io.SimulatedSource) else None)
        apply_b = Button(label='Applica parametri', button_type='success', width=180)
        apply_b.on_click(self._apply_params)
        save_b = Button(label='Salva configurazione', width=180)
        save_b.on_click(lambda: (self._apply_params(), self.cfg.to_json(),
                                 self.status(f'Configurazione salvata in {DEFAULT_CONFIG_FILE}')))
        load_b = Button(label='Carica configurazione', width=180)
        load_b.on_click(self._load_config)
        note = Div(width=900, text='''
<b>Note sui parametri</b><br>
• <b>UMA-16 su Windows</b>: per 16 canali installare il driver miniDSP UAC2/ASIO; il programma cerca
automaticamente un dispositivo con "UMA16"/"miniDSP" e ≥16 canali, preferendo ASIO.<br>
• <b>Banda utile</b>: con apertura di 12.6 cm la risoluzione sotto ~800 Hz è molto scarsa; sopra
c/(2·0.042) ≈ 4.1 kHz compaiono lobi di aliasing spaziale (attenuati dalla somma a banda larga). Per i droni
(frequenza di passaggio pala 100–400 Hz + armoniche fino a diversi kHz) la banda 0.8–6 kHz è un buon compromesso.<br>
• <b>Griglia emisfero (u,v)</b>: consigliata per droni (campo lontano, direzione azimut/elevazione);
<b>piano (x,y @ z)</b> come in bf_example_app, utile in laboratorio con sorgente a distanza nota.<br>
• L'array è piano: la <b>distanza</b> di una sorgente lontana non è stimabile, solo la direzione.''')
        self.param_layout = column(
            row(self.dev_select, ref, self.p['sample_rate']),
            row(self.p['replay_file'], self.rec_select),
            row(self.p['geometry'], self.invalid_mc),
            row(self.p['temperature_c'], self.p['humidity'], self.c_div, self.p['block_size'], self.p['window_s'],
                self.p['update_ms']),
            row(self.p['grid_mode'], self.p['grid_res'], self.p['max_off_axis_deg'], self.p['x_min'], self.p['x_max'],
                self.p['y_min'], self.p['y_max'], self.p['z'], self.p['far_field']),
            row(self.p['functional_gamma'], self.p['music_sources'], self.p['capon_loading']),
            row(self.p['sim_bpf_hz'], self.p['sim_snr_db'], self.p['sim_path'], self.sim_on),
            row(apply_b, save_b, load_b), note)
        self._refresh_recordings()

    def _refresh_recordings(self):
        files = sorted((str(p) for p in REC_DIR.glob('*.h5')), reverse=True)
        self.rec_select.options = [''] + files
        if hasattr(self, 'off_file'):
            self.off_file.options = [''] + files

    def _apply_params(self):
        cfg = self.cfg
        old_fs = cfg.sample_rate
        cfg.device_name = self.dev_select.value
        for k, w in self.p.items():
            if isinstance(w, CheckboxGroup):
                v = cb_val(w)
            else:
                v = w.value
            cur = getattr(cfg, k)
            if isinstance(cur, bool):
                v = bool(v)
            elif isinstance(cur, int):
                v = int(float(v))
            elif isinstance(cur, float):
                v = float(v)
            setattr(cfg, k, v)
        cfg.invalid_channels = sorted(int(v) for v in self.invalid_mc.value)
        self.c_div.text = f'c = {cfg.c:.1f} m/s'
        self.band_slider.end = cfg.sample_rate / 2 * 0.95
        cfg.f_max = min(cfg.f_max, self.band_slider.end)
        self.rebuild()
        if self.cb_id is not None:
            self.doc.remove_periodic_callback(self.cb_id)
            self.cb_id = self.doc.add_periodic_callback(self.tick, int(cfg.update_ms))
        msg = 'Parametri applicati.'
        if old_fs != cfg.sample_rate and self.source is not None:
            msg += ' Riavviare l\'acquisizione per la nuova frequenza di campionamento.'
        self.status(msg)

    def _load_config(self):
        self.cfg.__dict__.update(AppConfig.from_json().__dict__)
        for k, w in self.p.items():
            v = getattr(self.cfg, k)
            if isinstance(w, CheckboxGroup):
                w.active = [0] if v else []
            elif isinstance(w, (Select, TextInput)):
                w.value = str(v)
            else:
                w.value = v
        self.invalid_mc.value = [str(i) for i in self.cfg.invalid_channels]
        self._sync_noise_widgets()
        self.rebuild()
        self.status('Configurazione caricata.')

    # ================================================================== TAB OFFLINE (ACOULAR)
    def _build_offline(self):
        self.off_file = Select(title='Registrazione (.h5)', options=[], width=420)
        self.off_method = Select(title='Metodo acoular', value='BeamformerCleansc', width=220,
                                 options=['BeamformerBase', 'BeamformerFunctional', 'BeamformerCapon',
                                          'BeamformerEig', 'BeamformerMusic', 'BeamformerCleansc',
                                          'BeamformerClean', 'BeamformerDamas', 'BeamformerOrth'])
        self.off_freq = sp('Frequenza centrale (Hz)', 2000.0, 100, 100, 20000, width=150)
        self.off_num = Select(title='Banda', value='3', width=160,
                              options=[('0', 'singola riga'), ('1', 'ottava'), ('3', 'terzo d\'ottava')])
        self.off_start = sp('Inizio (s)', 0.0, 0.5, 0, width=90)
        self.off_len = sp('Durata (s)', 2.0, 0.5, 0.1, width=90)
        go = Button(label='Calcola (acoular)', button_type='primary', width=180)
        go.on_click(self._run_offline)
        self.off_div = Div(text='Seleziona una registrazione fatta con il pulsante "Registra" della scheda Live.',
                           width=700)
        self.off_cmap = LinearColorMapper(palette=viridis(100), low=-10, high=0, low_color=(0, 0, 0, 0),
                                          nan_color=(0, 0, 0, 0))
        self.off_src = ColumnDataSource(data=dict(image=[np.full((2, 2), -100.0)], x=[-1], y=[-1], dw=[2], dh=[2]))
        self.off_fig = figure(title='Mappa acoular (dB rispetto al massimo)', width=640, height=560,
                              x_range=Range1d(-1, 1), y_range=Range1d(-1, 1), tools='pan,wheel_zoom,reset,save,hover',
                              tooltips=[('dB', '@image')])
        self.off_fig.toolbar.logo = None
        self.off_fig.image(image='image', x='x', y='y', dw='dw', dh='dh', color_mapper=self.off_cmap,
                           source=self.off_src)
        self.off_fig.add_layout(ColorBar(color_mapper=self.off_cmap, title='dB'), 'right')
        self.off_fig.multi_line('xs', 'ys', source=self.horizon_src, line_color='#9e9e9e', line_dash='dotted')
        self.off_dyn = Slider(title='Dinamica (dB)', start=1, end=40, step=1, value=10, width=200)
        self.off_dyn.on_change('value', lambda a, o, n: setattr(self.off_cmap, 'low', -n))
        txt = Div(width=420, text='''
Questa scheda riprende la catena di <b>bf_example_app</b> (TimeSamples → PowerSpectra → SteeringVector →
Beamformer) usando <b>acoular</b> sulla stessa geometria, griglia, calibrazione dei guadagni e canali esclusi
dell'analisi live. È più lenta ma offre metodi di deconvoluzione (CleanSC, DAMAS...) utili per verificare
le prestazioni della localizzazione live su registrazioni con sorgente in posizione nota.''')
        self.off_layout = column(row(self.off_file, self.off_method, self.off_freq, self.off_num),
                                 row(self.off_start, self.off_len, go, self.off_dyn), self.off_div,
                                 row(self.off_fig, txt))

    def _run_offline(self):
        try:
            import acoular as ac
            ac.config.global_caching = 'none'
            if not self.off_file.value:
                self.off_div.text = 'Nessun file selezionato.'
                return
            data, fs = audio_io.load_multichannel(self.off_file.value)
            s0 = int(self.off_start.value * fs)
            data = data[s0:s0 + int(self.off_len.value * fs)]
            act = self.proc.active
            g = self.calib.gains[act] if self.cfg.use_gain_calibration else 1.0
            data = data[:, act] * g
            ts = ac.TimeSamples(data=data.astype(np.float64), sample_freq=fs)
            mg = ac.MicGeom(pos_total=self.proc.mpos)
            grid = self.proc.grid
            pts = grid.flat_points[grid.flat_valid].T
            if self.cfg.far_field:
                # campo lontano: punti a grande distanza lungo le stesse direzioni
                pts = pts / np.linalg.norm(pts, axis=0) * 1000.0
            ig = ac.ImportGrid(pos=pts)
            env = ac.Environment(c=self.cfg.c)
            st = ac.SteeringVector(grid=ig, mics=mg, env=env, steer_type='true level')
            ps = ac.PowerSpectra(source=ts, block_size=self.cfg.block_size, overlap='50%', window='Hanning')
            B = getattr(ac, self.off_method.value)
            kw = {}
            if self.off_method.value in ('BeamformerBase', 'BeamformerCleansc', 'BeamformerClean',
                                         'BeamformerDamas', 'BeamformerEig', 'BeamformerOrth', 'BeamformerCapon'):
                kw['r_diag'] = self.off_method.value not in ('BeamformerCapon',) and self.cfg.diag_removal
            if self.off_method.value == 'BeamformerFunctional':
                kw['gamma'] = self.cfg.functional_gamma
                kw['r_diag'] = False
            if self.off_method.value in ('BeamformerDamas', 'BeamformerClean'):
                kw['beamformer'] = ac.BeamformerBase(freq_data=ps, steer=st, r_diag=self.cfg.diag_removal)
                kw.pop('r_diag', None)
            bf = B(freq_data=ps, steer=st, **kw)
            t = time.time()
            res = bf.synthetic(float(self.off_freq.value), int(self.off_num.value))
            full = np.full(grid.flat_valid.shape, np.nan)
            full[grid.flat_valid] = np.ravel(res)
            m = full.reshape(grid.shape)
            ldb = 10 * np.log10(np.clip(m, 1e-30, None) / np.nanmax(m))
            ldb[~np.isfinite(ldb)] = -200
            s = grid.step
            x0, x1, y0, y1 = grid.extent
            self.off_src.data = dict(image=[ldb], x=[x0 - s / 2], y=[y0 - s / 2], dw=[x1 - x0 + s], dh=[y1 - y0 + s])
            self.off_fig.x_range.start, self.off_fig.x_range.end = x0, x1
            self.off_fig.y_range.start, self.off_fig.y_range.end = y0, y1
            k = int(np.nanargmax(m))
            iy, ix = np.unravel_index(k, grid.shape)
            az, el, _ = grid.to_angles(grid.x[ix], grid.y[iy])
            self.off_div.text = (f'{self.off_method.value}: calcolo in {time.time() - t:.1f} s. Picco in '
                                 f'({grid.x[ix]:+.2f}, {grid.y[iy]:+.2f}) → azimut {az:.1f}°, elevazione {el:.1f}°')
        except Exception as exc:
            traceback.print_exc()
            self.off_div.text = f'<span style="color:#b00020">Errore acoular: {exc}</span>'

    # ================================================================== TAB GUIDA
    def _build_guide(self):
        self.guide_layout = column(Div(width=1000, text=GUIDE_HTML))


GUIDE_HTML = '''
<h3>Flusso di lavoro consigliato</h3>
<ol>
<li><b>Parametri</b>: verificare il dispositivo (UMA-16), frequenza di campionamento 48 kHz, temperatura
(velocità del suono), tipo di griglia. Premere <i>Applica</i> e <i>Salva configurazione</i>.
Senza hardware si può provare tutto con la sorgente <b>Simulatore</b> (drone virtuale su traiettoria circolare).</li>
<li><b>Live</b>: avviare l'acquisizione e controllare spettri e mappa.</li>
<li><b>Calibrazione</b>: verifica canali → guadagni (altoparlante fisso sull'asse) → orientamento
(altoparlante fuori asse in posizione nota) → eventualmente SPL assoluto. I risultati sono salvati in
<code>data/calibration.json</code>.</li>
<li><b>Rumore / Ambiente</b>: sul campo, <i>senza</i> drone, eseguire l'<b>Autocalibrazione ambiente</b>:
registra il fondo, sceglie il preset (campo aperto, vento, foresta, canyon urbano) e la banda.</li>
<li><b>Addestramento</b>: registrare campioni "drone" riproducendo i suoni con l'altoparlante (fisso e in
movimento) e campioni delle classi di disturbo; addestrare e attivare il classificatore.</li>
<li><b>Live / Traiettoria</b>: con il drone (o l'altoparlante in movimento) osservare la mappa, la traccia filtrata,
azimut/elevazione nel tempo; esportare in CSV. Registrare le prove (pulsante <i>Registra</i>) per rianalizzarle
in modalità File o nella scheda <i>Analisi offline</i> con CleanSC.</li>
</ol>
<h3>Lettura della mappa</h3>
<p>Nella griglia <b>emisfero</b> il centro è la direzione perpendicolare all'array (zenit se l'array è rivolto al
cielo), il cerchio esterno è l'orizzonte. La croce rossa è il massimo istantaneo, il cerchio la stima filtrata dal
tracker, la linea bianca la traiettoria. Il piccolo schema dei microfoni al centro mostra l'orientamento
(numeri canale nella scheda Calibrazione).</p>
<h3>Limiti fisici dell'array</h3>
<p>Apertura 12.6 cm: larghezza del lobo principale ≈ λ/D (a 2 kHz circa 75°, a 5 kHz circa 30° con
beamforming convenzionale). Per questo sono disponibili metodi ad alta risoluzione (Funzionale, Capon, MUSIC)
e l'interpolazione sub-griglia del picco. Sopra ~4.1 kHz compaiono lobi di aliasing; la somma su banda larga
e la normalizzazione per frequenza li attenuano.</p>
'''


# ============================================================================== avvio
def server_doc(doc):
    DroneApp(doc)


if __name__ == '__main__':
    port = int(os.environ.get('UMA16_PORT', 5006))
    server = Server({'/': server_doc}, port=port, num_procs=1)
    server.start()
    print(f'UMA-16 Drone Locator su http://localhost:{port}/')
    server.io_loop.add_callback(server.show, '/')
    server.io_loop.start()
else:
    server_doc(curdoc())
