import os

# Sblocca il supporto ASIO su Windows
os.environ["SD_ENABLE_ASIO"] = "1"

import sounddevice as sd
import numpy as np
import wave

def main():
    print("==========================================")
    print("   Registrazione Singolo Mic da UMA-16   ")
    print("==========================================")

    # 1. Trova l'indice del driver ASIO miniDSP
    devices = sd.query_devices()
    target_device_idx = None
    
    for idx, dev in enumerate(devices):
        if "ASIO" in dev['name'] or "miniDSP" in dev['name']:
            # Verifica che sia associato all'host API ASIO
            if sd.query_hostapis(dev['hostapi'])['name'] == 'ASIO':
                target_device_idx = idx
                break

    if target_device_idx is None:
        print("[INFO] Driver miniDSP non rilevato automaticamente.")
        target_device_idx = int(input("Inserisci l'indice del driver miniDSP visto prima (es. 12): "))

    print(f"[OK] Utilizzo dispositivo [{target_device_idx}]: {devices[target_device_idx]['name']}")

    # 2. IMPOSTAZIONI DI CONFIGURAZIONE
    CHANNELS_IN = 16       # L'hardware richiede tassativamente 16 canali in ingresso
    SAMPLE_RATE = 48000    # Frequenza nativa della UMA-16
    BLOCK_SIZE = 1024
    
    # SCEGLI QUI IL MICROFONO DA SALVARE (da 0 a 15)
    # Canale 0 = Microfono 1, Canale 1 = Microfono 2, ecc.
    TARGET_MIC_CHANNEL = 0 
    
    WAV_FILENAME = "registrazione_mic1.wav"

    # 3. PREPARAZIONE DEL FILE WAV
    # Apriamo il file in modalità scrittura binaria ('wb')
    wav_file = wave.open(WAV_FILENAME, 'wb')
    wav_file.setnchannels(1)       # Salviamo 1 solo canale (Mono) per l'ascolto
    wav_file.setsampwidth(2)       # 2 byte = 16-bit PCM (formato audio standard compatibile ovunque)
    wav_file.setframerate(SAMPLE_RATE)

    # 4. CALLBACK PER LA CATTURA AUDIO NATIVA
    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        
        # Estraiamo solo la colonna del microfono scelto
        single_mic_data = indata[:, TARGET_MIC_CHANNEL]
        
        # Convertiamo i dati float in ingresso (-1.0 a 1.0) nel formato PCM 16-bit integer richiesto dal WAV
        audio_int16 = (single_mic_data * 32767.0).astype(np.int16)
        
        # Scriviamo i dati grezzi direttamente nel file sul disco
        wav_file.writeframes(audio_int16.tobytes())
        
        # Calcolo RMS per il feedback visivo in tempo reale
        rms = np.sqrt(np.mean(single_mic_data ** 2))
        print(f"Registrazione attiva su {WAV_FILENAME} | Livello RMS: {rms:.4f}", end="\r")

    # 5. AVVIO CATTURA
    try:
        print(f"\nAvvio dello streaming. Parla nel microfono...")
        with sd.InputStream(device=target_device_idx,
                            channels=CHANNELS_IN,
                            samplerate=SAMPLE_RATE,
                            blocksize=BLOCK_SIZE,
                            callback=audio_callback):
            print("Sto registrando... PREMI CTRL+C PER FERMARE E SALVARE IL FILE.\n")
            while True:
                sd.sleep(1000)
                
    except KeyboardInterrupt:
        print("\n\n[OK] Registrazione interrotta dall'utente.")
    finally:
        # Chiudiamo correttamente il file per salvare l'header WAV finale sul disco
        wav_file.close()
        print(f"[SUCCESSO] File audio salvato correttamente in: {os.path.abspath(WAV_FILENAME)}")

if __name__ == "__main__":
    main()
