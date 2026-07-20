# Radar Target Detection & Signal Analysis

A Python project that simulates radar return signals, applies digital filtering
for noise reduction, performs CFAR (Constant False Alarm Rate) target
detection, extracts signal features, and trains a machine-learning classifier
to distinguish true targets from false alarms.

## Tech Stack
- Python 3.9+
- NumPy
- SciPy
- Matplotlib
- Scikit-learn
- Pandas

## Project Structure
```
radar_target_detection/
├── src/
│   ├── signal_simulation.py    # Simulates radar returns (targets + clutter + noise)
│   ├── filtering.py             # Noise reduction / digital filters
│   ├── feature_extraction.py    # Peak detection & feature engineering
│   ├── detection.py             # CFAR detector + ML classifier
│   ├── visualization.py         # Plotting utilities
│   └── main.py                  # Runs the full end-to-end pipeline
├── output/                      # Generated plots & CSV report (created on run)
├── requirements.txt
└── README.md
```

## Quick Start
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/main.py
```

Outputs (plots + `detection_report.csv`) are written to the `output/` folder.

See **How_To_Run_And_Deploy.pdf** for full step-by-step setup, run, and
deployment instructions.
