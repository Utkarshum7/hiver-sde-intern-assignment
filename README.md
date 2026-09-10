# Hiver SDE Intern Assignment - AI Customer Support Agent

This repository contains an end-to-end AI Customer Support Agent pipeline built on the Kaggle *Customer Support on Twitter* dataset (`thoughtvector/customer-support-on-twitter`).

---

## Phase 1: Environment Setup & Dataset Profiling

### Requirements & Prerequisites
- **Python Version**: `Python 3.12` (or 3.11+)
- **Git**: Installed and initialized

### Environment Setup

1. **Clone & Navigate to Repository**:
   ```bash
   git clone <repo-url>
   cd Hiver-assignment
   ```

2. **Create and Activate Virtual Environment**:
   ```powershell
   # Windows PowerShell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Dataset Acquisition

> **Warning**: The raw Kaggle dataset (`twcs.csv`, ~500 MB - 1 GB) is **not committed** to Git and is excluded via `.gitignore`. Do not commit raw data files.

Place the raw dataset CSV under `data/raw/twcs.csv`.

### Option A: Kaggle API CLI (Recommended if configured)
If you have your Kaggle API key configured (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY` environment variables):
```bash
kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/raw/ --unzip
```

### Option B: Manual Download via Kaggle Web UI
1. Visit [Customer Support on Twitter on Kaggle](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter).
2. Download `customer-support-on-twitter.zip`.
3. Extract `twcs.csv` and place it in the project folder at:
   `data/raw/twcs.csv`

---

## Running Dataset Profiling

To inspect and profile the dataset memory-efficiently (using streaming chunks):

```bash
python scripts/profile_dataset.py --input data/raw/twcs.csv
```

### Running Unit Tests
To verify test suites and reusable profiler components:
```bash
pytest
```

---

## Expected Output Artifacts

Upon running `scripts/profile_dataset.py`, the following profile artifacts will be generated in `reports/`:

- `reports/data_profile.json`: Full machine-readable dataset statistics, message metrics, thread analysis, and brand candidate counts.
- `reports/data_profile.md`: Human-readable Markdown summary report.

---

## Repository Layout (Phase 1)
```
Hiver-assignment/
├── .gitignore                          # Exclusions for .venv, data/raw, cache, etc.
├── pytest.ini                          # Pytest configuration
├── README.md                           # Reproducibility & Phase 1 documentation
├── requirements.txt                    # Pinned Python dependencies
├── Hiver SDE Intern Assignment.pdf     # Original assignment prompt
├── data/
│   ├── raw/                            # Place raw twcs.csv here (git-ignored)
│   └── processed/                      # Filtered/processed brand data (git-ignored)
├── src/
│   ├── __init__.py
│   └── profiler.py                     # Streaming chunk profiler & schema parser
├── scripts/
│   └── profile_dataset.py              # CLI entry point to run profiling
├── reports/
│   ├── data_profile.json               # (Generated) Machine-readable metrics
│   └── data_profile.md                 # (Generated) Human-readable summary
└── tests/
    └── test_profiler.py                # Unit tests for profiler
```
