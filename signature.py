"""Impronta acustica: estrazione caratteristiche, addestramento e riconoscimento.

Caratteristiche (per finestra di analisi, dal segnale focalizzato sulla sorgente):
  - 36 energie in bande logaritmiche 80 Hz - 12 kHz, in dB, normalizzate (forma spettrale, non livello)
  - armonicita': punteggio del pettine armonico e frequenza fondamentale stimata (BPF dei rotori 60-500 Hz)
  - piattezza spettrale, centroide, rolloff 85 %, concentrazione dei picchi
Classificatore: RandomForest (scikit-learn) oppure, se non disponibile, centroidi gaussiani.
Le classi il cui nome contiene 'drone' sono considerate positive.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

EPS = 1e-20
N_BANDS = 36


def _band_edges(fs):
    return np.geomspace(80, min(12000, fs / 2.2), N_BANDS + 1)


def extract_features(psd, freqs):
    psd = np.asarray(psd, float) + EPS
    fs = 2 * freqs[-1]
    edges = _band_edges(fs)
    be = np.array([psd[(freqs >= lo) & (freqs < hi)].sum() if ((freqs >= lo) & (freqs < hi)).any()
                   else psd[np.argmin(abs(freqs - lo))] for lo, hi in zip(edges[:-1], edges[1:])])
    bdb = 10 * np.log10(be + EPS)
    shape = bdb - bdb.mean()
    b = (freqs > 150) & (freqs < min(10000, fs / 2.2))
    p = psd[b]
    f = freqs[b]
    flat = np.exp(np.mean(np.log(p))) / np.mean(p)
    centroid = np.log10(np.sum(f * p) / np.sum(p))
    cs = np.cumsum(p) / p.sum()
    rolloff = np.log10(f[min(np.searchsorted(cs, 0.85), len(f) - 1)])
    peak = np.sort(p)[-max(1, len(p) // 20):].sum() / p.sum()
    # armonicita': media dB sulle armoniche k*f0 - mediana locale
    db = 10 * np.log10(psd)
    from scipy.ndimage import median_filter
    rel = db - median_filter(db, 15, mode='nearest')
    df = freqs[1] - freqs[0]
    best, f0b = -np.inf, 0.0
    for f0 in np.arange(max(60, 2 * df), 500, max(2.0, df / 2)):
        ks = np.arange(1, 13) * f0
        ks = ks[ks < min(6000, fs / 2.2)]
        if len(ks) < 3:
            continue
        idx = np.round(ks / df).astype(int)
        s = np.mean(np.maximum(rel[idx], rel[np.clip(idx + 1, 0, len(rel) - 1)]))
        if s > best:
            best, f0b = s, f0
    return np.concatenate([shape, [flat, centroid, rolloff, peak, best, np.log10(f0b + 1)]])


class SignatureModel:
    def __init__(self):
        self.samples: dict[str, list] = {}
        self.class_psd: dict[str, list] = {}
        self.freqs = None
        self.clf = None
        self.classes = []
        self.report = 'Modello non addestrato.'
        self.prob_smooth = None

    # ------------------------------------------------------------ dati
    def add_sample(self, label, psd, freqs):
        label = label.strip() or 'senza_nome'
        self.samples.setdefault(label, []).append(extract_features(psd, freqs))
        self.class_psd.setdefault(label, []).append(10 * np.log10(np.asarray(psd) + EPS))
        self.freqs = freqs

    def counts(self):
        return {k: len(v) for k, v in self.samples.items()}

    def remove_class(self, label):
        self.samples.pop(label, None)
        self.class_psd.pop(label, None)

    def add_wav(self, label, data, fs_file, fs, block_size, window_s):
        """Aggiunge campioni da un file audio (mono o multicanale mediato)."""
        from math import gcd

        from scipy.signal import resample_poly
        x = np.asarray(data, float)
        if x.ndim > 1:
            x = x.mean(1)
        if fs_file != fs:
            g = gcd(int(fs), int(fs_file))
            x = resample_poly(x, fs // g, fs_file // g)
        n = int(window_s * fs)
        win = np.hanning(block_size)
        freqs = np.fft.rfftfreq(block_size, 1 / fs)
        added = 0
        for s in range(0, len(x) - n + 1, n // 2):
            seg = x[s:s + n]
            hop = block_size // 2
            nb = 1 + (len(seg) - block_size) // hop
            idx = np.arange(block_size)[None] + hop * np.arange(nb)[:, None]
            X = np.fft.rfft(seg[idx] * win, axis=1) * (np.sqrt(2) / win.sum())
            psd = np.mean(np.abs(X) ** 2, 0)
            if psd.sum() < 1e-14:
                continue
            self.add_sample(label, psd, freqs)
            added += 1
        return added

    # ------------------------------------------------------------ training
    def train(self):
        labels = [k for k, v in self.samples.items() if len(v) > 0]
        if len(labels) < 2:
            self.report = 'Servono almeno 2 classi (es. "drone" e "sfondo").'
            return self.report
        X = np.vstack([np.array(self.samples[k]) for k in labels])
        y = np.concatenate([[k] * len(self.samples[k]) for k in labels])
        self.classes = labels
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            self.clf = make_pipeline(StandardScaler(),
                                     RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                                            random_state=0))
            cv = min(5, min(len(self.samples[k]) for k in labels))
            acc = cross_val_score(self.clf, X, y, cv=cv).mean() if cv >= 2 else np.nan
            self.clf.fit(X, y)
            kind = 'RandomForest'
        except ImportError:
            self.clf = _GaussianCentroids().fit(X, y)
            acc, kind = np.nan, 'Centroidi gaussiani'
        cnt = ', '.join(f'{k}: {len(self.samples[k])}' for k in labels)
        self.report = f'Modello {kind} addestrato. Campioni -> {cnt}. Accuratezza (validazione incrociata): ' \
                      f'{acc * 100:.1f} %' if np.isfinite(acc) else f'Modello {kind} addestrato. Campioni -> {cnt}.'
        return self.report

    @property
    def trained(self):
        return self.clf is not None

    def positive(self, label):
        return 'drone' in label.lower()

    def predict(self, psd, freqs, smooth=0.6):
        """Restituisce (classe_piu_probabile, probabilita' drone, dict probabilita')."""
        if self.clf is None:
            return None, np.nan, {}
        f = extract_features(psd, freqs)[None]
        pr = self.clf.predict_proba(f)[0]
        cls = list(self.clf.classes_)
        probs = dict(zip(cls, pr))
        pdrone = float(sum(p for c, p in probs.items() if self.positive(c)))
        self.prob_smooth = pdrone if self.prob_smooth is None or not np.isfinite(self.prob_smooth) \
            else smooth * self.prob_smooth + (1 - smooth) * pdrone
        return cls[int(np.argmax(pr))], self.prob_smooth, probs

    def frequency_weights(self, freqs):
        """Pesi (0.05..1) per bin: dove lo spettro medio 'drone' supera le classi negative."""
        pos = [np.mean(v, 0) for k, v in self.class_psd.items() if self.positive(k) and v]
        neg = [np.mean(v, 0) for k, v in self.class_psd.items() if not self.positive(k) and v]
        if not pos or not neg or self.freqs is None:
            return None
        diff = np.mean(pos, 0) - np.mean(neg, 0)
        diff -= np.median(diff)
        w = np.clip(diff / 10, 0, 1) + 0.05
        from scipy.ndimage import uniform_filter1d
        w = uniform_filter1d(w, 5)
        return np.interp(freqs, self.freqs, w)

    # ------------------------------------------------------------ persistenza
    def save(self, path):
        with open(path, 'wb') as fh:
            pickle.dump({'samples': self.samples, 'class_psd': self.class_psd, 'freqs': self.freqs,
                         'clf': self.clf, 'classes': self.classes, 'report': self.report}, fh)

    def load(self, path):
        with open(path, 'rb') as fh:
            d = pickle.load(fh)
        self.__dict__.update(d)
        self.prob_smooth = None


class _GaussianCentroids:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.mu = np.array([X[y == c].mean(0) for c in self.classes_])
        self.sd = np.array([X[y == c].std(0) + 1e-3 for c in self.classes_])
        return self

    def predict_proba(self, X):
        ll = -0.5 * np.sum(((X[:, None] - self.mu[None]) / self.sd[None]) ** 2 + 2 * np.log(self.sd[None]), -1)
        ll -= ll.max(1, keepdims=True)
        p = np.exp(ll)
        return p / p.sum(1, keepdims=True)


def model_file() -> Path:
    from config import DATA_DIR
    return DATA_DIR / 'signature_model.pkl'
