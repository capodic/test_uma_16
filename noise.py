"""Profilo del rumore di fondo, aggiornamento adattivo e analisi automatica dell'ambiente.

Strategie di rimozione del rumore implementate nel motore:
  1. rimozione della diagonale della CSM  -> elimina rumore incoerente (vento, elettronica)
  2. sottrazione della CSM di rumore      -> elimina rumore coerente/stazionario (traffico, generatori)
  3. maschera SNR per banda              -> usa solo le frequenze in cui il segnale supera il fondo
  4. pesatura di stabilita' tonale        -> privilegia armoniche stabili (droni) rispetto a transitori
  5. pesatura con impronta addestrata     -> privilegia le bande caratteristiche del drone
  6. aggiornamento adattivo del fondo     -> il profilo segue il rumore quando non c'e' rilevamento
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

EPS = 1e-20


class NoiseModel:
    def __init__(self):
        self.C = None          # (F, 16, 16) CSM di rumore
        self.psd = None        # (F, 16)
        self.freqs = None
        self._acc = []
        self._frame_psd = []
        self.duration = 0.0
        self.report = ''
        self.suggestion = {}

    # ------------------------------------------------------------ stato
    def has_profile(self, nfreq=None):
        return self.C is not None and (nfreq is None or self.C.shape[0] == nfreq)

    def clear(self):
        self.__init__()

    def csm_band(self, band, active):
        return self.C[band][:, active][:, :, active]

    def trace_band(self, band, active):
        return np.real(np.einsum('fii->f', self.csm_band(band, active)))

    # ------------------------------------------------------------ registrazione
    def start_recording(self):
        self._acc, self._frame_psd, self.duration = [], [], 0.0

    def add_frame(self, C, freqs, seconds):
        self._acc.append(C)
        self._frame_psd.append(np.real(np.einsum('fii->fi', C)))
        self.freqs = freqs
        self.duration += seconds

    def finish_recording(self, mpos_all, c):
        if not self._acc:
            return 'Nessun dato registrato.'
        self.C = np.mean(self._acc, 0)
        self.psd = np.real(np.einsum('fii->fi', self.C))
        frames = np.array(self._frame_psd)            # (T, F, M)
        self.report, self.suggestion = analyse_environment(self.freqs, self.C, frames, mpos_all, c)
        self._acc = []
        return self.report

    def adapt(self, C, alpha):
        if self.C is None or self.C.shape != C.shape:
            return
        self.C = (1 - alpha) * self.C + alpha * C
        self.psd = np.real(np.einsum('fii->fi', self.C))

    # ------------------------------------------------------------ persistenza
    def save(self, path):
        if self.C is None:
            return
        np.savez_compressed(path, C=self.C, freqs=self.freqs, report=self.report)

    def load(self, path):
        d = np.load(path, allow_pickle=False)
        self.C = d['C']
        self.freqs = d['freqs']
        self.psd = np.real(np.einsum('fii->fi', self.C))
        self.report = str(d['report'])


def _pair_coherence(C, mpos, dist, tol=0.004):
    M = C.shape[1]
    pairs = [(i, j) for i in range(M) for j in range(i + 1, M)
             if abs(np.linalg.norm(mpos[:, i] - mpos[:, j]) - dist) < tol]
    if not pairs:
        return None
    coh = [np.abs(C[:, i, j]) ** 2 / (np.real(C[:, i, i] * C[:, j, j]) + EPS) for i, j in pairs]
    return np.mean(coh, 0)


def analyse_environment(freqs, C, frames, mpos, c):
    """Analizza il rumore di fondo e suggerisce preset e banda di analisi.

    Indicatori:
      - eccesso a bassa frequenza (< 300 Hz) e bassa coerenza tra microfoni vicini -> vento
      - alta coerenza (vicina a 1, oltre il campo diffuso) -> sorgenti di rumore direzionali (traffico)
      - forte variabilita' temporale in 2-8 kHz -> uccelli/insetti (transitori)
    """
    psd = np.mean(np.real(np.einsum('fii->fi', C)), 1)
    db = 10 * np.log10(psd + EPS)
    lo = (freqs > 40) & (freqs < 300)
    mid = (freqs > 1000) & (freqs < 4000)
    hi = (freqs > 2000) & (freqs < 8000)
    excess_lf = float(np.mean(db[lo]) - np.mean(db[mid])) if lo.any() and mid.any() else 0.0

    d = 0.042
    coh = _pair_coherence(C, mpos, d)
    k = 2 * np.pi * freqs * d / c
    diffuse = np.sinc(k / np.pi) ** 2
    band = (freqs > 300) & (freqs < 3000)
    coh_lo = float(np.mean(coh[(freqs > 40) & (freqs < 300)])) if coh is not None else np.nan
    coh_excess = float(np.mean(coh[band] - diffuse[band])) if coh is not None else 0.0

    frames_db = 10 * np.log10(np.mean(frames, 2) + EPS)      # (T, F)
    var_hi = float(np.mean(np.std(frames_db[:, hi], 0))) if frames_db.shape[0] > 2 else 0.0

    # f_min suggerita: prima frequenza (>400 Hz) oltre la quale il fondo scende entro 6 dB della mediana 1-6 kHz
    ref = np.median(db[(freqs > 1000) & (freqs < 6000)])
    cand = np.where((freqs > 400) & (db < ref + 6))[0]
    fmin = float(freqs[cand[0]]) if len(cand) else 800.0
    fmin = float(np.clip(np.ceil(fmin / 100) * 100, 500, 2000))

    lines = [f'Livello medio fondo 1-4 kHz: {np.mean(db[mid]):.1f} dB (rel.)',
             f'Eccesso bassa freq. (<300 Hz vs 1-4 kHz): {excess_lf:+.1f} dB',
             f'Coerenza microfoni adiacenti <300 Hz: {coh_lo:.2f} (campo diffuso ~1, vento << 1)',
             f'Eccesso di coerenza 0.3-3 kHz rispetto al campo diffuso: {coh_excess:+.2f}',
             f'Variabilita\' temporale 2-8 kHz: {var_hi:.1f} dB']
    if excess_lf > 15 and coh_lo < 0.6:
        preset = 'Campo aperto ventoso'
        why = 'forte rumore a bassa frequenza e poco coerente: tipico del vento'
    elif coh_excess > 0.15:
        preset = 'Canyon urbano'
        why = 'rumore coerente/direzionale (traffico, macchinari, riflessioni)'
    elif var_hi > 4:
        preset = 'Foresta'
        why = 'forti transitori in alta frequenza (uccelli, insetti, fruscio)'
    elif excess_lf > 8:
        preset = 'Campo aperto'
        why = 'moderato rumore di vento'
    else:
        preset = 'Campo aperto'
        why = 'fondo debole e poco strutturato'
    lines.append(f'=> Ambiente suggerito: {preset} ({why}); f_min suggerita ~{fmin:.0f} Hz')
    return '\n'.join(lines), {'preset': preset, 'f_min': fmin}


def noise_file(name='noise_profile.npz') -> Path:
    from config import DATA_DIR
    return DATA_DIR / name
