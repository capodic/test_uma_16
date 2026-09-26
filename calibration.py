"""Calibrazione dell'array: verifica canali, guadagni, fase, orientamento, livello assoluto.

Procedura consigliata (vedi README):
 1. Verifica canali      - pochi secondi con rumore ambiente o sorgente accesa: canali morti/rumorosi
 2. Guadagni (e fase)    - altoparlante fisso in posizione nota (meglio sull'asse, >= 1.5 m) con rumore rosa
 3. Orientamento         - stessa sorgente in posizione nota fuori asse (es. a destra): sceglie tra
                           8 combinazioni di ribaltamento/scambio assi quella coerente con la realta'
 4. Livello assoluto     - inserendo l'SPL letto con un fonometro vicino all'array
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from geometry import ORIENTATIONS, FocusGrid, transform_positions
from processing import steering

EPS = 1e-20


class Calibration:
    def __init__(self, nch=16):
        self.nch = nch
        self.gains = np.ones(nch)
        self.phase = None           # (F, nch) complesso unitario
        self.phase_block = None
        self.report = ''

    def reset(self):
        self.__init__(self.nch)

    def save(self, path):
        d = {'gains': self.gains.tolist(), 'report': self.report}
        if self.phase is not None:
            d['phase_re'] = self.phase.real.tolist()
            d['phase_im'] = self.phase.imag.tolist()
            d['phase_block'] = self.phase_block
        Path(path).write_text(json.dumps(d), encoding='utf-8')

    def load(self, path):
        d = json.loads(Path(path).read_text(encoding='utf-8'))
        self.gains = np.array(d['gains'])
        self.report = d.get('report', '')
        if 'phase_re' in d:
            self.phase = np.array(d['phase_re']) + 1j * np.array(d['phase_im'])
            self.phase_block = d.get('phase_block')


def band_power(C, freqs, fmin=200, fmax=8000):
    b = (freqs >= fmin) & (freqs <= fmax)
    return np.real(np.einsum('fii->i', C[b]))


def check_channels(C, freqs):
    """Rileva canali morti, saturi/rumorosi o scorrelati. Restituisce (report, invalid, livelli_dB)."""
    p = band_power(C, freqs)
    db = 10 * np.log10(p + EPS)
    med = np.median(db)
    b = (freqs > 300) & (freqs < 1500)
    M = C.shape[1]
    coh = np.zeros(M)
    for i in range(M):
        cc = [np.mean(np.abs(C[b, i, j]) ** 2 / (np.real(C[b, i, i] * C[b, j, j]) + EPS))
              for j in range(M) if j != i]
        coh[i] = np.mean(cc)
    invalid, lines = [], []
    for i in range(M):
        tag = 'ok'
        if db[i] < med - 15:
            tag = 'MORTO / molto basso'
            invalid.append(i)
        elif db[i] > med + 10:
            tag = 'RUMOROSO / alto'
            invalid.append(i)
        elif coh[i] < 0.3 * np.median(coh):
            tag = 'scorrelato dagli altri'
            invalid.append(i)
        lines.append(f'ch {i:2d}: {db[i] - med:+6.1f} dB rispetto alla mediana, coerenza media {coh[i]:.2f}  {tag}')
    if not invalid:
        lines.append('Tutti i canali risultano validi.')
    else:
        lines.append(f'Canali da escludere suggeriti: {invalid}')
    return '\n'.join(lines), invalid, db - med


def gain_calibration(C, freqs, mpos, source_pos, fmin=500, fmax=6000, invalid=()):
    """Guadagni per canale che equalizzano la potenza misurata con quella attesa (1/r^2)."""
    p = band_power(C, freqs, fmin, fmax)
    r = np.linalg.norm(np.asarray(source_pos)[:, None] - mpos, axis=0)
    expected = 1 / r ** 2
    ratio = expected / (p + EPS)
    valid = np.array([i not in set(invalid) for i in range(len(p))])
    ratio /= np.median(ratio[valid])
    g = np.sqrt(ratio)
    g[~valid] = 1.0
    g = np.clip(g, 0.25, 4.0)
    lines = [f'ch {i:2d}: guadagno {g[i]:.3f} ({20 * np.log10(g[i]):+.2f} dB)' for i in range(len(g))]
    spread = 20 * np.log10(g[valid]).std()
    lines.append(f'Dispersione guadagni: {spread:.2f} dB (tipico MEMS: < 1 dB)')
    return g, '\n'.join(lines)


def phase_calibration(C, freqs, mpos, source_pos, c, min_coh=0.8):
    """Correzione di fase per canale e frequenza (sorgente in posizione nota, campo vicino)."""
    M = C.shape[1]
    diag = np.real(np.einsum('fii->fi', C))
    ref = int(np.argsort(diag[(freqs > 500) & (freqs < 5000)].mean(0))[M // 2])
    a = steering(freqs, np.asarray(source_pos, float)[None], mpos, c, far_field=False)[:, 0, :]
    E = a / a[:, [ref]]
    H = C[:, :, ref] / (C[:, [ref], ref] + EPS)
    coh = np.abs(C[:, :, ref]) ** 2 / (diag * diag[:, [ref]] + EPS)
    corr = E / (H + EPS)
    corr = corr / (np.abs(corr) + EPS)
    ph = np.unwrap(np.angle(corr), axis=0)
    from scipy.ndimage import median_filter
    ph = median_filter(ph, size=(9, 1), mode='nearest')
    good = coh > min_coh
    ph[~good] = 0.0
    out = np.exp(1j * ph)
    band = (freqs > 500) & (freqs < 6000)
    resid = np.degrees(np.sqrt(np.mean(ph[band] ** 2, 0)))
    lines = [f'ch {i:2d}: correzione fase RMS {resid[i]:5.1f} gradi' for i in range(M)]
    lines.append(f'Canale di riferimento: {ref}. Frequenze con coerenza < {min_coh}: non corrette.')
    return out, '\n'.join(lines)


def find_orientation(C, freqs, pos_raw, grid: FocusGrid, expected_xy, c, far_field, fmin=1000, fmax=5000,
                     active=None):
    """Prova le 8 orientazioni della geometria e restituisce quella con il picco piu' vicino all'atteso."""
    band = np.where((freqs >= fmin) & (freqs <= fmax))[0][::2]
    active = np.arange(pos_raw.shape[1]) if active is None else np.asarray(active)
    Cb = C[band][:, active][:, :, active].copy()
    M = len(active)
    Cb[:, np.arange(M), np.arange(M)] = 0
    pts = grid.flat_points[grid.flat_valid]
    gxy = np.stack([grid.X.ravel(), grid.Y.ravel()], 1)[grid.flat_valid]
    res = []
    for o in ORIENTATIONS:
        pos = transform_positions(pos_raw, *o)[:, active]
        A = steering(freqs[band], pts, pos, c, far_field) / M
        P = np.real(np.sum(np.matmul(A.conj(), Cb) * A, -1))
        P = (P / P.max(1, keepdims=True)).sum(0)
        k = np.argmax(P)
        err = float(np.hypot(*(gxy[k] - np.asarray(expected_xy))))
        res.append((err, o, gxy[k]))
    res.sort(key=lambda r: r[0])
    lines = [f'flip_x={o[0]!s:5} flip_y={o[1]!s:5} swap_xy={o[2]!s:5} -> picco ({p[0]:+.2f},{p[1]:+.2f}) '
             f'errore {e:.3f}' for e, o, p in res]
    return res[0][1], '\n'.join(lines)
