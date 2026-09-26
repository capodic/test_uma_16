"""Tracciamento della sorgente: filtro di Kalman a velocita' costante con gate di associazione."""

from __future__ import annotations

import csv
import time

import numpy as np


class Tracker:
    def __init__(self, gate=0.5, q=0.5, r=0.03, max_missed=8):
        self.gate = gate
        self.q = q
        self.r = r
        self.max_missed = max_missed
        self.reset()

    def reset(self):
        self.x = None          # [px, py, vx, vy]
        self.P = None
        self.t = None
        self.missed = 0
        self.history = []      # dict per punto
        self.raw = []

    @property
    def active(self):
        return self.x is not None

    def _predict(self, dt):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
        Q = G @ G.T * self.q ** 2
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z, detected, t=None, extra=None):
        """z = (gx, gy) misura; detected = bool. Restituisce lo stato filtrato (gx, gy) o None."""
        t = time.time() if t is None else t
        extra = extra or {}
        if detected:
            self.raw.append((t, z[0], z[1]))
        if self.x is None:
            if detected:
                self.x = np.array([z[0], z[1], 0.0, 0.0])
                self.P = np.diag([self.r, self.r, 1.0, 1.0])
                self.t = t
                self.missed = 0
                self._log(t, extra)
                return self.x[:2]
            return None
        dt = max(1e-3, t - self.t)
        self.t = t
        self._predict(dt)
        if detected:
            H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
            R = np.eye(2) * self.r ** 2
            y = np.asarray(z) - H @ self.x
            S = H @ self.P @ H.T + R
            d = float(np.sqrt(y @ np.linalg.solve(S, y)))
            if np.hypot(*y) <= self.gate or d < 3:
                K = self.P @ H.T @ np.linalg.inv(S)
                self.x = self.x + K @ y
                self.P = (np.eye(4) - K @ H) @ self.P
                self.missed = 0
            else:
                # misura fuori gate (riflessione / seconda sorgente): conta come mancata
                self.missed += 1
                if self.missed > self.max_missed:   # ri-inizializza sul nuovo bersaglio
                    self.x = np.array([z[0], z[1], 0.0, 0.0])
                    self.P = np.diag([self.r, self.r, 1.0, 1.0])
                    self.missed = 0
        else:
            self.missed += 1
            if self.missed > self.max_missed:
                self.x = None
                return None
        self._log(t, extra)
        return self.x[:2]

    def _log(self, t, extra):
        rec = {'t': t, 'x': float(self.x[0]), 'y': float(self.x[1]),
               'vx': float(self.x[2]), 'vy': float(self.x[3]), 'missed': self.missed}
        rec.update(extra)
        self.history.append(rec)

    def export_csv(self, path):
        if not self.history:
            return 0
        keys = list(self.history[0].keys())
        for h in self.history:
            for k in h:
                if k not in keys:
                    keys.append(k)
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(self.history)
        return len(self.history)
