# Camera Analyzer - DroidCam + OpenRouter

Programma Python per scattare foto da DroidCam e analizzarle con OpenRouter API per identificare la risposta corretta a domande visualizzate sullo schermo.

## Requisiti

- Python 3.7+
- DroidCam installato e attivo sul dispositivo mobile
- API Key di OpenRouter

## Installazione

1. Installa le dipendenze:
```bash
pip install -r requirements.txt
```

2. Crea il file `.env` con le tue configurazioni:
```bash
cp env.example .env
```

Poi modifica il file `.env` con i tuoi valori:
```env
DROIDCAM_IP=192.168.1.17
DROIDCAM_PORT=4747
OPENROUTER_API_KEY=sk-or-v1-la-tua-chiave-api
OPENROUTER_MODEL=openai/gpt-4o
```

**Nota:** Il file `.env` è già incluso nel `.gitignore` per sicurezza.

Alternativamente, puoi usare variabili d'ambiente:
```bash
export DROIDCAM_IP="192.168.1.17"
export DROIDCAM_PORT="4747"
export OPENROUTER_API_KEY="sk-..."
export OPENROUTER_MODEL="openai/gpt-4o"
```

## Configurazione DroidCam

1. Installa DroidCam sul tuo dispositivo mobile
2. Avvia DroidCam sul dispositivo
3. Connetti il dispositivo alla stessa rete WiFi del computer
4. Nota l'IP mostrato nell'app DroidCam
5. Aggiorna `DROIDCAM_IP` nel codice o nella variabile d'ambiente

## Utilizzo

Esegui il programma:
```bash
python camera_analyzer.py
```

Il programma:
- Si connetterà a DroidCam
- Attenderà che tu premi INVIO per scattare una foto
- Invierà l'immagine a OpenRouter API con il prompt per identificare la risposta corretta
- Mostrerà il risultato (numero della risposta corretta)

Premi 'q' e INVIO per uscire.

## Modelli Supportati

Il programma usa modelli con supporto vision. Modelli consigliati:
- `openai/gpt-4o` (default)
- `openai/gpt-4-vision-preview`
- `google/gemini-pro-vision`
- Altri modelli vision supportati da OpenRouter

## Note

- L'immagine viene salvata come `last_capture.jpg` per debug
- Assicurati che DroidCam sia attivo prima di eseguire il programma
- La qualità dell'analisi dipende dalla chiarezza dell'immagine catturata

