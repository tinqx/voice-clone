# Voice-Clone Evaluation (Tortoise TTS & OpenVoice)

Objektive und praxisnahe Bewertung von Voice-Cloning-Ausgaben (Eigenaufnahmen → Synthese → Metriken → HomePod-Tests). 

**Implementierungsbasis:** https://github.com/natlamir/tortoise-WebUI

**Vergleich:** https://github.com/myshell-ai/OpenVoice/tree/main/openvoice

---

## Inhalt
- `eval_voiceclone.py` – berechnet MCD-DTW-SL, SECS, f0-RMSE, f0-Korrelation, Dauer-Diff. für ein Paar (Original vs. Synthese)
- `tts_evaluation.py` - erzeugt Tortoise TTS Audios aus Text 
- `benchmark.py` – helper für mehrere Konstellationen
- `create_radar.py` – erzeugt Radar-Plots aus den JSON-Ergebnissen
- `*_results.json` – Beispielergebnisse (kurz/lang, Tortoise/OpenVoice)
- `*_radar.png` – Radar-Visualisierungen zu den Ergebnissen

---

## Beispiel Kommandozeile: 
```bash
python tts_evaluation.py --text "Hallo HomePod" --voice myvoice --preset high_quality
