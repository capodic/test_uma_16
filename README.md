# UMA-16 Drone Locator

Applicazione Bokeh (derivata dalla struttura di `bf_example_app` di spectacoular) per localizzare e
tracciare in tempo reale una sorgente acustica, in particolare un drone, con l'array **miniDSP UMA-16**
(16 microfoni MEMS su una griglia 4×4 con passo 42 mm).

## Installazione (Windows)

1. Installare il **driver miniDSP UMA-16** (UAC2/ASIO) dal sito miniDSP. Senza driver, Windows può
   esporre solo 2 canali.
2. Python 3.10–3.12, poi, dalla cartella del progetto:
   ```
   pip install -r requirements.txt
   ```
3. Avvio: doppio clic su `avvia_windows.bat`, oppure `python main.py`, oppure `bokeh serve --show main.py`.
   Si apre il browser su http://localhost:5006/.

Il programma cerca automaticamente un dispositivo il cui nome contiene "UMA16" o "miniDSP" con almeno
16 canali, preferendo ASIO. Se non lo trova, selezionarlo in **Parametri → Dispositivo di ingresso**.
Senza hardware si può provare tutto con la sorgente **Simulatore**: un drone virtuale con armoniche
della frequenza di passaggio pala, vento e cinguettii.

## Struttura

| File | Contenuto |
|---|---|
| `main.py` | Interfaccia Bokeh (schede Live, Traiettoria, Calibrazione, Rumore/Ambiente, Addestramento, Parametri, Analisi offline, Guida) |
| `processing.py` | FFT/CSM, beamforming a banda larga (Convenzionale, Funzionale, Capon/MVDR, MUSIC), indice di direzionalità, spettro focalizzato |
| `audio_io.py` | Acquisizione UMA-16 (sounddevice), riproduzione di file h5/wav, simulatore, registrazione HDF5 compatibile con acoular |
| `geometry.py` | Geometria UMA-16 (xml acoular o tabella interna), orientamento, griglie emisfero (u,v) e piano (x,y @ z) |
| `calibration.py` | Verifica canali, guadagni, fase, orientamento automatico, SPL assoluto |
| `noise.py` | Profilo del rumore di fondo, aggiornamento adattivo, analisi automatica dell'ambiente |
| `signature.py` | Impronta acustica: caratteristiche spettrali e armoniche, RandomForest, pesi di frequenza |
| `tracking.py` | Filtro di Kalman a velocità costante con gate ed esportazione CSV |
| `config.py` | Tutti i parametri e i preset ambientali |

I dati (configurazione, calibrazione, profilo di rumore, modello, registrazioni, traiettorie) sono salvati
nella cartella `data/`.

**Perché non si usa direttamente la catena acoular per il live:** la catena di `bf_example_app`
(TimeSamples → PowerSpectra → SteeringVector → Beamformer) lavora su file e usa una cache. Per
aggiornamenti ogni 250 ms il motore live è scritto in numpy vettorizzato, con le stesse convenzioni
di acoular. La catena acoular originale è mantenuta nella scheda **Analisi offline**, con CleanSC,
DAMAS e altri metodi, per verificare le registrazioni.

## Procedura di messa a punto

1. **Parametri**: indicare la temperatura (da cui si calcola la velocità del suono), la griglia
   (*Emisfero* per i droni) e la banda. Premere *Applica* e poi *Salva*.
2. **Live → Avvia acquisizione**: controllare che tutti i canali abbiano segnale (Spettro → *Tutti i canali*).
3. **Calibrazione**, con un altoparlante che riproduce rumore rosa:
   - *Verifica canali*: segnala i canali morti, rumorosi o scorrelati; *Applica canali suggeriti* li esclude.
   - *Guadagni*: altoparlante **fisso** in asse (x=0, y=0, z ≥ 1,5 m). La calibrazione di *fase* è opzionale.
   - *Orientamento*: altoparlante fuori asse in una posizione nota (es. x=+1, y=0, z=2). Il programma prova
     le 8 combinazioni di ribaltamento/scambio degli assi e sceglie quella corretta. Questo risolve
     l'ambiguità nota di acoular tra `minidsp_uma-16.xml` e `_mirrored.xml` (dipende da quale lato
     guarda l'array).
   - *SPL*: fonometro accanto all'array, inserire il valore letto.
4. **Rumore / Ambiente** (sul campo, **senza drone**): *Autocalibrazione ambiente*.
5. **Addestramento**:
   - registrare la classe `drone` riproducendo i suoni con l'altoparlante, sia fisso sia in movimento;
   - registrare le classi negative (`sfondo`, `uccelli`, `traffico`, `voci`...). Si possono anche
     caricare file WAV;
   - premere *Addestra* e poi attivare *Usa il classificatore* ed eventualmente, nella scheda Rumore,
     *Pesatura frequenze con impronta*.
6. **Prove con l'altoparlante in movimento**: osservare la traccia nella mappa e nella scheda
   **Traiettoria** (azimut ed elevazione nel tempo), poi esportare in CSV. Con *Registra (h5)* la prova
   viene salvata. Può essere rianalizzata con la sorgente *File* o nella scheda *Analisi offline*.

## Rilevamento

Una sorgente è considerata **rilevata** quando valgono tutte le condizioni che si applicano:

- l'**indice di direzionalità** (rapporto tra l'autovalore dominante della CSM e la media degli altri,
  nella banda) supera la soglia. Questo indice non dipende dal metodo di beamforming: il contrasto della
  mappa di MUSIC o Capon è sempre alto, anche con solo rumore;
- l'**SNR** del segnale focalizzato rispetto al profilo di fondo supera la soglia (se il profilo esiste);
- la **P(drone)** del classificatore supera la soglia (se il classificatore è attivo).

## Rimozione del rumore di fondo

| Ambiente | Rumore tipico | Contromisure (preset) |
|---|---|---|
| Campo aperto | vento: turbolenza incoerente tra i microfoni, sotto 300–500 Hz | rimozione della diagonale CSM, f_min 600–900 Hz, cuffia antivento, media temporale |
| Foresta | fruscio diffuso e uccelli/insetti (transitori modulati tra 2 e 8 kHz) | sottrazione della CSM di fondo, pesatura di stabilità tonale, classe "uccelli" nel classificatore, Capon |
| Canyon urbano | traffico coerente a bassa frequenza, riflessioni sulle facciate | sottrazione CSM adattiva, f_min ≥ 1 kHz, MUSIC/Capon, gate stretto nel tracker |
| Interno | riverbero | frequenze alte, finestre brevi, verifica offline con CleanSC |

L'autocalibrazione misura:

- l'eccesso di livello sotto i 300 Hz;
- la coerenza tra microfoni adiacenti, confrontata con il campo diffuso teorico sinc²(kd);
- la variabilità temporale tra 2 e 8 kHz.

Da queste misure sceglie il preset e la f_min. Con *aggiornamento adattivo* il profilo di fondo segue
lentamente l'ambiente, ma solo quando non c'è rilevamento e l'SNR è basso.

## Limiti fisici

- L'apertura di 12,6 cm dà una risoluzione angolare limitata: il lobo principale del beamforming
  convenzionale è largo circa 75° a 2 kHz e circa 30° a 5 kHz. Per questo sono disponibili i metodi
  Funzionale, Capon e MUSIC e l'interpolazione del picco tra i punti della griglia.
- Sopra c/(2·d) ≈ 4,1 kHz compare l'aliasing spaziale; la somma su banda larga lo attenua.
- L'array è piano: si stima la **direzione** (azimut ed elevazione), non la distanza di una sorgente lontana.
  Per la posizione 3D servono due array distanziati (triangolazione).
- Tempo di elaborazione: circa 100–250 ms per aggiornamento con i valori predefiniti. Se il PC è lento,
  aumentare il *passo griglia* o l'*aggiornamento (ms)*, oppure restringere la banda.

## Problemi comuni

- **Solo 2 canali o dispositivo non trovato**: installare il driver ASIO miniDSP e usare sounddevice
  ≥ 0.4.6. La variabile `SD_ENABLE_ASIO=1` è già impostata dal programma.
- **La sorgente appare speculare**: eseguire *Determina orientamento* oppure spuntare flip x/y.
- **Rilevamenti falsi**: registrare di nuovo il fondo, alzare la soglia di direzionalità o di SNR,
  attivare il classificatore.
