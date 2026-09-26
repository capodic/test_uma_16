"""Motore di elaborazione in tempo reale: FFT/CSM, beamforming a banda larga, spettri.

L'elaborazione e' vettorizzata in numpy (piu' leggera della catena acoular basata su cache, che resta
disponibile nella scheda 'Analisi offline'). Le convenzioni sono quelle di acoular:
    C(f) = E[x x^H],  h = a / M,  P = h^H C h  (potenza della sorgente riferita al centro array).
"""

from __future__ import annotations

import numpy as np

from geometry import FocusGrid, load_positions

METHODS = ['Convenzionale', 'Funzionale', 'Capon (MVDR)', 'MUSIC']
EPS = 1e-20


def steering(freqs, points, mpos, c, far_field):
    """Vettori di steering a[f, g, m] (complex64)."""
    f = np.asarray(freqs)[:, None, None]
    if far_field:
        d = points / np.linalg.norm(points, axis=1, keepdims=True)
        proj = d @ mpos                                   # (G, M)  u . p_m
        return np.exp(2j * np.pi * f * proj[None] / c).astype(np.complex64)
    r = np.linalg.norm(points[:, :, None] - mpos[None], axis=1)   # (G, M)
    r0 = np.linalg.norm(points, axis=1)[:, None]
    return ((r0 / r)[None] * np.exp(-2j * np.pi * f * (r - r0)[None] / c)).astype(np.complex64)


def csm_from_blocks(x, block_size, window=None, overlap=0.5):
    """x: (N, M) -> spettri X (B, F, M) e CSM (F, M, M). Normalizzazione a potenza per bin."""
    n, m = x.shape
    if window is None:
        window = np.hanning(block_size)
    hop = max(1, int(block_size * (1 - overlap)))
    nb = 1 + (n - block_size) // hop if n >= block_size else 0
    if nb <= 0:
        return None, None
    idx = np.arange(block_size)[None] + hop * np.arange(nb)[:, None]
    blocks = x[idx] * window[None, :, None]               # (B, N, M)
    X = np.fft.rfft(blocks, axis=1) * (np.sqrt(2) / window.sum())   # (B, F, M), |X|^2 ~ potenza tono
    C = np.einsum('bfm,bfn->fmn', X, X.conj(), optimize=True) / nb
    return X, C


class Processor:
    def __init__(self, cfg, calibration=None, noise=None):
        self.cfg = cfg
        self.calib = calibration
        self.noise = noise
        self.signature_weights = None   # (F,) opzionale dal modulo impronta
        self.build()

    # ------------------------------------------------------------------ setup
    def build(self):
        cfg = self.cfg
        self.fs = cfg.sample_rate
        self.c = cfg.c
        self.pos_all = load_positions(cfg.geometry, cfg.flip_x, cfg.flip_y, cfg.swap_xy)
        nall = self.pos_all.shape[1]
        self.active = np.array([i for i in range(nall) if i not in set(int(k) for k in cfg.invalid_channels)])
        self.mpos = self.pos_all[:, self.active]
        self.M = len(self.active)
        self.bs = int(cfg.block_size)
        self.window = np.hanning(self.bs)
        self.freqs = np.fft.rfftfreq(self.bs, 1 / self.fs)
        self.band = np.where((self.freqs >= cfg.f_min) & (self.freqs <= cfg.f_max))[0]
        if len(self.band) == 0:
            self.band = np.array([np.argmin(abs(self.freqs - 1000))])
        self.grid = FocusGrid(cfg.grid_mode, cfg.grid_res, cfg.x_min, cfg.x_max, cfg.y_min, cfg.y_max,
                              cfg.z, cfg.max_off_axis_deg)
        self.gidx = np.where(self.grid.flat_valid)[0]
        pts = self.grid.flat_points[self.gidx]
        self.A = steering(self.freqs[self.band], pts, self.mpos, self.c, cfg.far_field)  # (Fb, G, M)
        self.C_s = None
        self.hist = []          # storia spettri banda per pesatura stabilita'

    # ------------------------------------------------------------------ helpers
    def channel_correction(self, idx):
        """Correzione complessa (F, len(idx)) da calibrazione guadagno/fase."""
        corr = np.ones((len(self.freqs), len(idx)), np.complex128)
        if self.calib is None:
            return corr
        if self.cfg.use_gain_calibration and self.calib.gains is not None:
            corr *= self.calib.gains[idx][None, :]
        if self.cfg.use_phase_calibration and self.calib.phase is not None \
                and self.calib.phase.shape[0] == len(self.freqs):
            corr *= self.calib.phase[:, idx]
        return corr

    def spectra(self, x, all_channels=False, calibrated=True):
        """Solo FFT/CSM con calibrazione applicata (usato anche da calibrazione e rumore)."""
        idx = np.arange(self.pos_all.shape[1]) if all_channels else self.active
        x = np.asarray(x, np.float64)[:, idx]
        X, C = csm_from_blocks(x, self.bs, self.window)
        if X is None:
            return None, None
        if not calibrated:
            return X, C
        corr = self.channel_correction(idx)
        X = X * corr[None]
        C = C * (corr[:, :, None] * corr.conj()[:, None, :])
        return X, C

    @staticmethod
    def _psd_fix(C):
        """Proietta CSM (F,M,M) sul cono semidefinito positivo (dopo sottrazione rumore)."""
        w, v = np.linalg.eigh(C)
        w = np.clip(w, 0, None)
        return (v * w[:, None, :]) @ v.conj().transpose(0, 2, 1)

    def _beamform(self, C, method):
        A = self.A
        M = self.M
        cfg = self.cfg
        if method == 'Convenzionale':
            Cb = C.copy()
            if cfg.diag_removal:
                idx = np.arange(M)
                Cb[:, idx, idx] = 0
            h = A / M
            T = np.matmul(h.conj(), Cb.astype(np.complex64))   # (F, G, M)
            P = np.real(np.sum(T * h, -1))
            if cfg.diag_removal:
                P *= M / (M - 1)
            return np.clip(P, EPS, None)
        w, v = np.linalg.eigh(C)                          # autovalori crescenti
        w = np.clip(w, 0, None)
        if method == 'Funzionale':
            g = max(1.0, float(cfg.functional_gamma))
            Cg = (v * (w ** (1 / g))[:, None, :]) @ v.conj().transpose(0, 2, 1)
            h = A / np.sqrt(M)
            P = np.real(np.sum(np.matmul(h.conj(), Cg.astype(np.complex64)) * h, -1))
            return np.clip(P, EPS, None) ** g / M
        if method == 'Capon (MVDR)':
            tr = np.real(np.trace(C, axis1=1, axis2=2))[:, None, None] / M
            Ci = np.linalg.inv(C + cfg.capon_loading * tr * np.eye(M)[None] + EPS * np.eye(M)[None])
            q = np.real(np.sum(np.matmul(A.conj(), Ci.astype(np.complex64)) * A, -1))
            return 1.0 / np.clip(q, EPS, None)
        if method == 'MUSIC':
            k = int(np.clip(cfg.music_sources, 1, M - 1))
            En = v[:, :, : M - k]                          # sottospazio rumore
            proj = np.matmul(A.conj(), En.astype(np.complex64))   # (F, G, M-k)
            q = np.sum(np.abs(proj) ** 2, -1) / M
            return 1.0 / np.clip(q, 1e-12, None)
        raise ValueError(method)

    def _bin_weights(self, C_band, psd_sum_band):
        w = np.ones(len(self.band))
        cfg = self.cfg
        if self.noise is not None and self.noise.has_profile(len(self.freqs)):
            tn = self.noise.trace_band(self.band, self.active)
            ts = np.real(np.trace(C_band, axis1=1, axis2=2))
            snr = 10 * np.log10((ts + EPS) / (tn + EPS))
            mask = snr > cfg.snr_mask_db
            if mask.sum() >= 3:
                w *= mask
        if cfg.stability_weighting:
            db = 10 * np.log10(psd_sum_band + EPS)
            self.hist.append(db)
            self.hist = self.hist[-8:]
            if len(self.hist) >= 3:
                st = np.std(np.array(self.hist), 0)
                w *= np.exp(-st / 3.0)
            from scipy.ndimage import median_filter
            tonal = np.clip(db - median_filter(db, 9, mode='nearest'), 0, 12) / 12 + 0.1
            w *= tonal
        if cfg.signature_weighting and self.signature_weights is not None \
                and len(self.signature_weights) == len(self.freqs):
            w *= self.signature_weights[self.band]
        if w.sum() <= 0:
            w = np.ones(len(self.band))
        return w

    def focused_spectrum(self, X, gx, gy):
        """Spettro del segnale focalizzato (delay-and-sum) verso il punto di griglia (gx, gy)."""
        p = self._grid_point(gx, gy)[None]
        a = steering(self.freqs, p, self.mpos, self.c, self.cfg.far_field)[:, 0, :]   # (F, M)
        y = np.einsum('bfm,fm->bf', X, a.conj()) / self.M
        return np.mean(np.abs(y) ** 2, 0)

    def _grid_point(self, gx, gy):
        g = self.grid
        if g.mode.startswith('Emisfero'):
            r = min(np.hypot(gx, gy), 0.9999)
            return np.array([gx, gy, np.sqrt(1 - r ** 2)]) * g.z
        return np.array([gx, gy, g.z])

    # ------------------------------------------------------------------ elaborazione
    def process(self, x):
        cfg = self.cfg
        X_all, C_all = self.spectra(x, all_channels=True)
        if X_all is None:
            return None
        act = self.active
        X = X_all[:, :, act]
        C = C_all[:, act][:, :, act]
        psd = np.mean(np.abs(X) ** 2, 0)                 # (F, M)
        a = float(np.clip(cfg.csm_smoothing, 0, 0.98))
        if self.C_s is None or self.C_s.shape != C.shape:
            self.C_s = C
        else:
            self.C_s = a * self.C_s + (1 - a) * C
        Cb = self.C_s[self.band]
        if cfg.csm_subtraction and self.noise is not None and self.noise.has_profile(len(self.freqs)):
            Cn = self.noise.csm_band(self.band, self.active)
            Cb = self._psd_fix(Cb - cfg.csm_sub_factor * Cn)
        w = self._bin_weights(Cb, psd[self.band].mean(1))
        # indice di direzionalita': autovalore dominante / media degli altri (indipendente dal metodo)
        ev = np.clip(np.linalg.eigvalsh(Cb), 0, None)
        dir_idx = 10 * np.log10(ev[:, -1] / (ev[:, :-1].mean(1) + EPS) + EPS)
        dir_index = float(np.sum(dir_idx * w) / w.sum())
        Pf = self._beamform(Cb, cfg.method)             # (Fb, G)
        if cfg.freq_norm:
            Pf = Pf / Pf.max(1, keepdims=True)
        P = np.sum(Pf * w[:, None], 0) / w.sum()
        full = np.full(self.grid.shape[0] * self.grid.shape[1], np.nan)
        full[self.gidx] = P
        mapv = full.reshape(self.grid.shape)
        k = int(np.nanargmax(mapv))
        iy, ix = np.unravel_index(k, self.grid.shape)
        gx, gy = float(self.grid.x[ix]), float(self.grid.y[iy])
        pmax = float(np.nanmax(mapv))
        contrast = 10 * np.log10(pmax / (np.nanmedian(mapv) + EPS) + EPS)
        # sub-grid refinement (interpolazione parabolica)
        gx, gy = self._refine(mapv, ix, iy, gx, gy)
        focused = self.focused_spectrum(X, gx, gy)
        band_level = 10 * np.log10(np.sum(focused[self.band]) + EPS) + cfg.spl_offset_db
        snr_band = np.nan
        if self.noise is not None and self.noise.has_profile(len(self.freqs)):
            nb = np.mean(self.noise.psd[self.band][:, self.active], 1).sum() / self.M
            snr_band = 10 * np.log10(np.sum(focused[self.band]) / (nb + EPS) + EPS)
        az, el, th = self.grid.to_angles(gx, gy)
        return dict(
            map_db=10 * np.log10(mapv / pmax + EPS),
            peak=(gx, gy), contrast_db=contrast, dir_index_db=dir_index, level_db=band_level, snr_band_db=snr_band,
            az=az, el=el, theta=th, freqs=self.freqs, psd=psd, focused=focused,
            C=C, C_all=C_all, X=X, weights=w,
        )

    def _refine(self, m, ix, iy, gx, gy):
        g = self.grid
        try:
            if 0 < ix < m.shape[1] - 1:
                l, c0, r = m[iy, ix - 1], m[iy, ix], m[iy, ix + 1]
                d = (l - r) / (2 * (l - 2 * c0 + r))
                if np.isfinite(d) and abs(d) < 1:
                    gx += d * g.step
            if 0 < iy < m.shape[0] - 1:
                l, c0, r = m[iy - 1, ix], m[iy, ix], m[iy + 1, ix]
                d = (l - r) / (2 * (l - 2 * c0 + r))
                if np.isfinite(d) and abs(d) < 1:
                    gy += d * g.step
        except Exception:
            pass
        return gx, gy
