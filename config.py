"""Parametri dell'applicazione (salvabili/caricabili in JSON)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)
DEFAULT_CONFIG_FILE = DATA_DIR / 'config.json'


def speed_of_sound(temp_c: float, rel_humidity: float = 50.0) -> float:
    """Velocita' del suono [m/s] in funzione della temperatura (e, debolmente, dell'umidita')."""
    c = 331.3 * (1.0 + temp_c / 273.15) ** 0.5
    return c + 0.006 * rel_humidity  # correzione umidita' approssimata (~ +0.6 m/s al 100 %)


@dataclass
class AppConfig:
    # --- acquisizione -------------------------------------------------------
    source: str = 'UMA-16 (live)'          # 'UMA-16 (live)' | 'File (h5/wav)' | 'Simulatore'
    device_name: str = ''                  # vuoto = ricerca automatica; oppure ID, es. '1'
    use_asio: bool = False                 # abilita ASIO (serve il driver ASIO miniDSP; riavvio app)
    sample_rate: int = 48000
    num_channels: int = 16
    replay_file: str = ''
    # --- geometria ------------------------------------------------------------
    geometry: str = 'minidsp_uma-16_corrected.xml'   # file in geometries/, 'acoular: <file>' o 'interna ...'
    flip_x: bool = False
    flip_y: bool = False
    swap_xy: bool = False
    invalid_channels: list = field(default_factory=list)
    # --- ambiente -----------------------------------------------------------
    temperature_c: float = 20.0
    humidity: float = 50.0
    # --- analisi ------------------------------------------------------------
    block_size: int = 1024
    window_s: float = 0.5                  # durata finestra di analisi
    update_ms: int = 250                   # periodo aggiornamento
    f_min: float = 800.0
    f_max: float = 6000.0
    method: str = 'Convenzionale'          # Convenzionale | Funzionale | Capon (MVDR) | MUSIC
    functional_gamma: float = 8.0
    music_sources: int = 1
    capon_loading: float = 0.05
    freq_norm: bool = True                 # normalizza ogni banda prima della somma
    # --- griglia ------------------------------------------------------------
    grid_mode: str = 'Emisfero (u,v)'      # 'Emisfero (u,v)' | 'Piano (x,y @ z)'
    grid_res: float = 0.04                 # passo (u,v adimensionale oppure metri)
    x_min: float = -2.0
    x_max: float = 2.0
    y_min: float = -2.0
    y_max: float = 2.0
    z: float = 3.0                         # distanza piano (m) o distanza nominale (emisfero)
    far_field: bool = True
    max_off_axis_deg: float = 80.0         # emisfero: limite angolo dall'asse
    # --- rumore / ambiente --------------------------------------------------
    environment: str = 'Campo aperto'
    diag_removal: bool = True
    csm_subtraction: bool = False
    csm_sub_factor: float = 1.0
    adaptive_noise: bool = True
    adaptive_alpha: float = 0.01
    csm_smoothing: float = 0.5             # media esponenziale CSM tra aggiornamenti (0 = nessuna)
    snr_mask_db: float = 3.0               # usa solo bande con SNR > soglia (se profilo rumore presente)
    stability_weighting: bool = False      # privilegia componenti tonali stabili (anti uccelli/transienti)
    signature_weighting: bool = False      # pesa le frequenze con l'impronta addestrata
    # --- rilevamento / tracking ---------------------------------------------
    detect_threshold_db: float = 8.0       # indice di direzionalita' (dB) minimo
    snr_detect_db: float = 6.0             # SNR minimo rispetto al profilo di fondo (se presente)
    use_classifier: bool = False
    classifier_threshold: float = 0.6
    track_gate: float = 0.3                # distanza massima (unita' griglia) per associazione
    track_process_noise: float = 0.5
    track_max_missed: int = 8
    trail_length: int = 400
    # --- calibrazione -------------------------------------------------------
    spl_offset_db: float = 0.0
    use_gain_calibration: bool = True
    use_phase_calibration: bool = False
    # --- simulatore ---------------------------------------------------------
    sim_bpf_hz: float = 180.0
    sim_snr_db: float = 10.0
    sim_path: str = 'Cerchio'              # Cerchio | Fisso | Linea

    @property
    def c(self) -> float:
        return speed_of_sound(self.temperature_c, self.humidity)

    def to_json(self, path: Path | str = DEFAULT_CONFIG_FILE) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding='utf-8')

    @classmethod
    def from_json(cls, path: Path | str = DEFAULT_CONFIG_FILE) -> 'AppConfig':
        cfg = cls()
        p = Path(path)
        if p.exists():
            data = json.loads(p.read_text(encoding='utf-8'))
            names = {f.name for f in fields(cls)}
            for k, v in data.items():
                if k in names:
                    setattr(cfg, k, v)
            cfg._migrate_geometry()
        return cfg

    def _migrate_geometry(self):
        """Le configurazioni salvate con il file acoular 'minidsp_uma-16.xml' (x specchiate) o con la
        vecchia tabella interna passano alla geometria corretta; i ribaltamenti vanno rifatti."""
        old = {'minidsp_uma-16.xml', 'interna (UMA-16)'}   # nomi usati dalle versioni precedenti
        if self.geometry in old:
            self.geometry = 'minidsp_uma-16_corrected.xml'
            self.flip_x = self.flip_y = self.swap_xy = False
            self.geometry_migrated = True
        elif not self.geometry.startswith(('acoular: ', 'interna')) and '/' not in self.geometry \
                and '\\' not in self.geometry and not (APP_DIR / 'geometries' / self.geometry).exists():
            self.geometry = 'acoular: ' + self.geometry   # altro file acoular salvato senza prefisso


# Preset ambientali: valori suggeriti per la riduzione del rumore di fondo.
ENVIRONMENT_PRESETS = {
    'Campo aperto': dict(
        f_min=600.0, f_max=6000.0, diag_removal=True, csm_subtraction=False, adaptive_noise=True,
        csm_smoothing=0.5, stability_weighting=False, method='Funzionale', functional_gamma=8.0,
        detect_threshold_db=8.0,
        note='Rumore dominante: vento (incoerente tra microfoni, < 300 Hz). '
             'Rimozione diagonale CSM + f_min alta. Usare antivento in spugna sull\'array.'),
    'Campo aperto ventoso': dict(
        f_min=900.0, f_max=6000.0, diag_removal=True, csm_subtraction=False, adaptive_noise=True,
        csm_smoothing=0.7, stability_weighting=True, method='Funzionale', functional_gamma=8.0,
        detect_threshold_db=9.0,
        note='Vento forte: alzare f_min, maggiore media temporale, pesatura stabilita\' tonale.'),
    'Foresta': dict(
        f_min=700.0, f_max=5000.0, diag_removal=True, csm_subtraction=True, adaptive_noise=True,
        csm_smoothing=0.6, stability_weighting=True, method='Capon (MVDR)', functional_gamma=8.0,
        detect_threshold_db=9.0,
        note='Fruscio foglie (diffuso) + uccelli/insetti (transitori a 2-8 kHz, modulati in frequenza). '
             'Sottrazione CSM rumore + pesatura stabilita\' (le armoniche del drone sono stabili). '
             'Consigliato il classificatore addestrato con suoni di uccelli come classe negativa.'),
    'Canyon urbano': dict(
        f_min=1000.0, f_max=6000.0, diag_removal=True, csm_subtraction=True, adaptive_noise=True,
        csm_smoothing=0.5, stability_weighting=False, method='MUSIC', functional_gamma=8.0,
        detect_threshold_db=10.0,
        note='Traffico coerente a bassa frequenza e riflessioni sulle facciate (sorgenti immagine). '
             'Sottrazione CSM adattiva, f_min >= 1 kHz, MUSIC/Capon; il tracker scarta i salti '
             'dovuti a riflessioni (gate stretto).'),
    'Interno / laboratorio': dict(
        f_min=1000.0, f_max=7000.0, diag_removal=True, csm_subtraction=True, adaptive_noise=False,
        csm_smoothing=0.3, stability_weighting=False, method='Convenzionale', functional_gamma=8.0,
        detect_threshold_db=8.0,
        note='Riverbero: usare frequenze alte, finestre brevi, analisi offline con CleanSC per verifica.'),
    'Personalizzato': dict(note='Parametri impostati manualmente.'),
}
