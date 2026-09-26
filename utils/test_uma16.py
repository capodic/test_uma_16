import os

# FONDAMENTALE: Dice a sounddevice di sbloccare le funzioni ASIO su Windows
os.environ["SD_ENABLE_ASIO"] = "1"

import sounddevice as sd
import numpy as np

def main():
    print("==========================================")
    print("   Scansione Dispositivi ASIO in Python   ")
    print("==========================================")

    # 1. Elenca le API Host per essere sicuri che ASIO sia attivo
    host_apis = sd.query_hostapis()
    asio_available = any(api['name'] == 'ASIO' for api in host_apis)
    
    if not asio_available:
        print("[ERRORE] Il supporto ASIO non e' attivo. Verifica l'installazione dei driver miniDSP.")
        return

    # 2. Cerca l'indice esatto della UMA-16 nella lista dispositivi
    devices = sd.query_devices()
    target_device_idx = None
    target_device_name = ""

    print("\nDispositivi rilevati:")
    for idx, dev in enumerate(devices):
        # Filtra solo i dispositivi associati all'host ASIO
        if dev['hostapi'] == next(i for i, api in enumerate(host_apis) if api['name'] == 'ASIO'):
            print(f" [{idx}] {dev['name']} (Canali Input: {dev['max_input_channels']})")
            if "miniDSP" in dev['name'] or "UMA" in dev['name']:
                target_device_idx = idx
                target_device_name = dev['name']

    # Se non trova il nome miniDSP, ti permette comunque di forzare l'indice manualmente
    if target_device_idx is None:
        print("\n[INFO] Driver miniDSP non trovato col nome standard.")
        val = input("Inserisci l'indice numerico del dispositivo che vuoi usare (es. 0): ")
        target_device_idx = int(val)
        target_device_name = devices[target_device_idx]['name']

    print(f"\n[OK] Configurato su dispositivo [{target_device_idx}]: {target_device_name}")

    # 3. IMPOSTAZIONI STREAMING LIVE MICROFONI
    CHANNELS = 16        # La UMA-16 richiede tassativamente 16 canali paralleli
    SAMPLE_RATE = 48000  # Frequenza hardware nativa del miniDSP
    BLOCK_SIZE = 512     # Campioni per blocco (piu' e' basso, minore e' la latenza)

    # La funzione di Callback viene chiamata automaticamente ogni volta che arrivano nuovi dati
    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        
        # 'indata' e' una matrice NumPy a due dimensioni: (512 righe, 16 colonne)
        # Ogni colonna corrisponde al flusso audio live di uno specifico microfono
        mic_1_stream = indata[:, 0]   # Canale 0 (Microfono 1)
        mic_16_stream = indata[:, 15] # Canale 15 (Microfono 16)
        
        # Test: Calcoliamo l'ampiezza RMS in tempo reale del primo microfono
        rms = np.sqrt(np.mean(mic_1_stream ** 2))
        print(f"Livello Mic 1 (RMS Live): {rms:.4f} | Ricezione 16 canali attiva...", end="\r")

    # 4. AVVIO DELLO STREAMING
    try:
        print("\nAvvio dello stream in corso...")
        with sd.InputStream(device=target_device_idx,
                            channels=CHANNELS,
                            samplerate=SAMPLE_RATE,
                            blocksize=BLOCK_SIZE,
                            callback=audio_callback):
            print("Streaming avviato con successo! Premi CTRL+C per fermare il programma.\n")
            while True:
                sd.sleep(1000) # Mantiene attivo il thread principale
                
    except KeyboardInterrupt:
        print("\nStreaming terminato dall'utente.")
    except Exception as e:
        print(f"\n[ERRORE STREAMS]: {e}")

if __name__ == "__main__":
    main()
