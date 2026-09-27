"""Sorgenti audio multicanale: UMA-16 live (sounddevice), file (h5 acoular / wav), simulatore.

Tutte le sorgenti producono blocchi float32 di forma (n_campioni, n_canali) in una coda.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path

import numpy as np

# Su Windows la UMA-16 espone 16 canali tramite il driver miniDSP UAC2 (MME / WASAPI / DirectSound)
# e, se installato il driver ASIO, anche via ASIO. Il supporto ASIO di sounddevice (>= 0.5) va abilitato
# PRIMA dell'import: lo si attiva solo se richiesto nella configurazione (use_asio), perche' se il driver
# ASIO non e' caricabile l'apertura del dispositivo fallisce ("Failed to load ASIO driver").
def _asio_requested() -> bool:
    try:
        import json
        from config import DEFAULT_CONFIG_FILE
        return bool(json.loads(DEFAULT_CONFIG_FILE.read_text(encoding='utf-8')).get('use_asio', False))
    except Exception:
        return False


if _asio_requested():
    os.environ['SD_ENABLE_ASIO'] = '1'

try:
    import sounddevice as sd
except Exception as exc:  # PortAudio assente
    sd = None
    _SD_ERROR = str(exc)
else:
    _SD_ERROR = ''

UMA_KEYWORDS = ('uma16', 'uma-16', 'uma 16', 'minidsp', 'mchstreamer')
# ordine di preferenza delle API audio di Windows (ASIO per ultimo: richiede il suo driver)
API_PRIORITY = ('wasapi', 'mme', 'directsound', 'wdm-ks', 'asio')

BLOCK_MS = 50


# ----------------------------------------------------------------------------- dispositivi
def list_input_devices(min_channels: int = 1) -> list[str]:
    if sd is None:
        return []
    out = []
    hostapis = sd.query_hostapis()
    for i, d in enumerate(sd.query_devices()):
        if d['max_input_channels'] >= min_channels:
            api = hostapis[d['hostapi']]['name']
            out.append(f"{i}: {d['name']} [{api}] ({d['max_input_channels']} ch)")
    return out


def device_id(text: str) -> int | None:
    """'1', '#1' oppure '1: Linea (UMA16v2) [MME] (16 ch)' -> 1."""
    t = str(text).strip().lstrip('#')
    if not t:
        return None
    try:
        return int(t.split(':')[0].strip())
    except ValueError:
        return None


def uma16_candidates(preferred: str = '', min_channels: int = 16) -> list[int]:
    """Dispositivi da provare, in ordine. Se l'utente ne ha indicato uno, solo quello."""
    if sd is None:
        return []
    pid = device_id(preferred)
    if pid is not None:
        return [pid]
    hostapis = sd.query_hostapis()
    cands = []
    for i, d in enumerate(sd.query_devices()):
        name = d['name'].lower()
        if d['max_input_channels'] >= min_channels and any(k in name for k in UMA_KEYWORDS):
            api = hostapis[d['hostapi']]['name'].lower()
            prio = next((k for k, a in enumerate(API_PRIORITY) if a in api), len(API_PRIORITY))
            cands.append((prio, i))
    return [i for _, i in sorted(cands)]


def find_uma16(preferred: str = '') -> int | None:
    c = uma16_candidates(preferred)
    return c[0] if c else None


# ----------------------------------------------------------------------------- sorgenti
class BaseSource:
    name = 'base'

    def __init__(self, fs: int, nch: int = 16):
        self.fs = int(fs)
        self.nch = nch
        self.q: queue.Queue = queue.Queue(maxsize=400)
        self.running = False
        self.error = ''
        self.info = ''
        self.overflows = 0

    def _push(self, block: np.ndarray):
        try:
            self.q.put_nowait(block.astype(np.float32, copy=False))
        except queue.Full:
            self.overflows += 1

    def read_all(self) -> np.ndarray | None:
        blocks = []
        while True:
            try:
                blocks.append(self.q.get_nowait())
            except queue.Empty:
                break
        return np.concatenate(blocks, 0) if blocks else None

    def start(self):
        raise NotImplementedError

    def stop(self):
        self.running = False


class LiveSource(BaseSource):
    name = 'UMA-16 (live)'

    def __init__(self, fs=48000, nch=16, device: str = ''):
        super().__init__(fs, nch)
        self.device = device
        self.stream = None

    def start(self):
        if sd is None:
            raise RuntimeError(f'sounddevice/PortAudio non disponibile: {_SD_ERROR}')
        cands = uma16_candidates(self.device)
        if not cands:
            raise RuntimeError('UMA-16 non trovata automaticamente: scegliere il dispositivo (es. "1: ...") '
                               'nel menu "Dispositivo" della scheda Live o Parametri.')
        errors = []
        for idx in cands:
            try:
                dev = sd.query_devices(idx)
            except Exception as exc:
                errors.append(f'#{idx}: {exc}')
                continue
            api = sd.query_hostapis(dev['hostapi'])['name']
            nch = min(self.nch, dev['max_input_channels'])
            if nch < 1:
                errors.append(f"#{idx} {dev['name']} [{api}]: nessun canale di ingresso")
                continue
            rates = [self.fs] + [r for r in (int(dev['default_samplerate']),) if r != self.fs]
            for fs in rates:
                try:
                    self._open(idx, nch, fs, api)
                    self.fs = fs
                    self.nch = nch
                    self.running = True
                    warn = '' if nch >= 16 else f' - ATTENZIONE: solo {nch} canali'
                    self.info = f"#{idx} {dev['name']} [{api}] - {nch} ch @ {fs} Hz{warn}"
                    if errors:
                        self.info += f' (scartati: {"; ".join(errors)})'
                    return
                except Exception as exc:
                    errors.append(f"#{idx} {dev['name']} [{api}] @ {fs} Hz: {exc}")
        raise RuntimeError('nessun dispositivo utilizzabile. Tentativi: ' + ' | '.join(errors))

    def _open(self, idx, nch, fs, api):
        def cb(indata, frames, t, status):
            if status:
                self.overflows += 1
            self._push(indata.copy())

        extra = None
        if 'wasapi' in api.lower():
            try:
                extra = sd.WasapiSettings(auto_convert=True)
            except Exception:
                extra = None
        stream = sd.InputStream(device=idx, channels=nch, samplerate=fs, dtype='float32',
                                blocksize=int(fs * BLOCK_MS / 1000), callback=cb, extra_settings=extra)
        stream.start()
        self.stream = stream

    def stop(self):
        super().stop()
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            finally:
                self.stream = None


class _ThreadSource(BaseSource):
    """Sorgente che genera blocchi in un thread a velocita' tempo-reale."""

    def start(self):
        self.running = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        n = int(self.fs * BLOCK_MS / 1000)
        t_next = time.perf_counter()
        while self.running:
            try:
                self._push(self.generate(n))
            except Exception as exc:  # pragma: no cover
                self.error = str(exc)
                self.running = False
                break
            t_next += n / self.fs
            dt = t_next - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
            else:
                t_next = time.perf_counter()

    def generate(self, n):
        raise NotImplementedError


def load_multichannel(path: str) -> tuple[np.ndarray, int]:
    p = Path(path)
    if p.suffix.lower() in ('.h5', '.hdf5'):
        import h5py
        with h5py.File(p, 'r') as f:
            ds = f['time_data']
            return np.asarray(ds[:], np.float32), int(ds.attrs['sample_freq'])
    import soundfile as sf
    data, fs = sf.read(str(p), dtype='float32', always_2d=True)
    return data, int(fs)


class FileSource(_ThreadSource):
    name = 'File (h5/wav)'

    def __init__(self, path: str, nch=16, loop=True):
        data, fs = load_multichannel(path)
        super().__init__(fs, min(nch, data.shape[1]))
        self.data = data[:, :self.nch]
        self.pos = 0
        self.loop = loop
        self.info = f'{Path(path).name}: {data.shape[1]} ch, {len(data) / fs:.1f} s @ {fs} Hz'

    def generate(self, n):
        out = np.empty((n, self.nch), np.float32)
        k = 0
        while k < n:
            m = min(n - k, len(self.data) - self.pos)
            out[k:k + m] = self.data[self.pos:self.pos + m]
            k += m
            self.pos += m
            if self.pos >= len(self.data):
                if not self.loop:
                    self.running = False
                    return out[:k]
                self.pos = 0
        return out


class SimulatedSource(_ThreadSource):
    """Drone simulato (armoniche della frequenza di passaggio pala + rumore motore) in movimento,
    con rumore incoerente (vento) e, opzionalmente, cinguettii da una direzione fissa."""

    name = 'Simulatore'

    def __init__(self, positions: np.ndarray, fs=48000, c=343.0, bpf=180.0, snr_db=10.0,
                 path='Cerchio', birds=True, seed=0):
        super().__init__(fs, positions.shape[1])
        self.mpos = positions
        self.c = c
        self.bpf = bpf
        self.snr = 10 ** (snr_db / 20)
        self.path = path
        self.birds = birds
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.harm_amp = 1.0 / np.arange(1, 31) ** 0.7
        self.harm_phase = self.rng.uniform(0, 2 * np.pi, 30)
        self.info = f'Drone simulato BPF={bpf:.0f} Hz, traiettoria {path}'
        self.true_pos = np.array([0, 0, 10.0])
        self.drone_on = True
        self._ref_rms = None

    def source_position(self, t):
        if self.path == 'Fisso':
            return np.array([4.0, 2.0, 10.0])
        if self.path == 'Linea':
            x = -15 + (t * 2.0) % 30
            return np.array([x, 3.0, 10.0])
        w = 2 * np.pi / 20.0
        return np.array([8 * np.cos(w * t), 8 * np.sin(w * t), 10.0])

    def _delays(self, p):
        r = np.linalg.norm(p[:, None] - self.mpos, axis=0)
        r0 = np.linalg.norm(p)
        return (r - r0) / self.c, r0

    def generate(self, n):
        t = self.t + np.arange(n) / self.fs
        p = self.source_position(self.t)
        self.true_pos = p
        tau, r0 = self._delays(p)
        amp = 10.0 / r0
        f0 = self.bpf * (1 + 0.01 * np.sin(2 * np.pi * 0.3 * self.t))
        out = np.zeros((n, self.nch))
        tt = t[:, None] - tau[None, :]
        for k, (a, ph) in enumerate(zip(self.harm_amp, self.harm_phase), 1):
            if k * f0 > self.fs / 2.2:
                break
            out += a * np.cos(2 * np.pi * k * f0 * tt + ph)
        # rumore larga banda del motore, ritardato in frequenza
        nb = self.rng.standard_normal(n) * 0.3
        spec = np.fft.rfft(nb)
        f = np.fft.rfftfreq(n, 1 / self.fs)
        spec = spec * (f > 500)
        out += np.fft.irfft(spec[:, None] * np.exp(-2j * np.pi * f[:, None] * tau[None, :]), n, axis=0)
        out *= amp * 0.05
        # rumore incoerente (vento/elettronico), livello riferito al drone (anche se spento)
        if self._ref_rms is None:
            self._ref_rms = np.sqrt(np.mean(out ** 2)) + 1e-12
        sig_rms = self._ref_rms
        if not self.drone_on:
            out[:] = 0
        noise = self.rng.standard_normal((n, self.nch))
        lf = np.cumsum(self.rng.standard_normal((n, self.nch)), 0)
        lf -= lf.mean(0)
        lf /= (lf.std() + 1e-9)
        out += sig_rms / self.snr * (noise + 1.5 * lf)
        # cinguettii occasionali (direzione fissa)
        if self.birds and self.rng.random() < 0.08:
            pb = np.array([-6.0, 5.0, 3.0])
            taub, _ = self._delays(pb)
            fc = self.rng.uniform(3000, 5000)
            tl = (t - self.t)[:, None] - taub[None, :]
            ch = np.sin(2 * np.pi * (fc * tl + 20000 * tl ** 2))
            env = np.hanning(n)[:, None]
            out += 2 * sig_rms * ch * env
        self.t += n / self.fs
        return out.astype(np.float32)


# ----------------------------------------------------------------------------- registrazione
class Recorder:
    """Registra i dati grezzi in HDF5 compatibile con acoular (dataset 'time_data')."""

    def __init__(self, path: Path, fs: int, nch: int):
        import h5py
        self.path = Path(path)
        self.f = h5py.File(self.path, 'w')
        self.ds = self.f.create_dataset('time_data', shape=(0, nch), maxshape=(None, nch),
                                        dtype='float32', chunks=(4096, nch))
        self.ds.attrs['sample_freq'] = float(fs)
        self.n = 0

    def write(self, block: np.ndarray):
        m = len(block)
        self.ds.resize(self.n + m, axis=0)
        self.ds[self.n:self.n + m] = block
        self.n += m

    def close(self):
        self.f.close()
