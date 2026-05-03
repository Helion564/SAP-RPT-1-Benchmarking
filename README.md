# SAP RPT-1 Benchmarking

## 🚀 Setup

### Option 1: Docker (Recommended for Reproducibility)

```bash
# Clone the repo
git clone <repo-url>
cd "MINI proj SAP"

# Copy .env.example to .env and paste your HuggingFace token
cp .env.example .env

# Build containers
docker-compose build

# Run SAP RPT-1 experiment
docker-compose run sap-rpt1 -m runners.run_experiment --dataset analcatdata_authorship --model sap-rpt1-hf

# Run baselines batch
docker-compose run baselines -m runners.run_batch --datasets config/datasets.yaml --models config/models.yaml
```

### Option 2: Local Install (Python >= 3.11 required)

```bash
# Clone the repo
git clone <repo-url>
cd "MINI proj SAP"

# Install everything in one command
pip install -e ".[models,baselines]"

# Download datasets (19 datasets from OpenML)
cd code
python -m datasets.download_tabarena
cd ..
```

## 🔑 Hugging Face Token Setup (Required for SAP RPT-1 OSS)

The SAP RPT-1 OSS model weights are **gated** on Hugging Face:

1. Create account at [huggingface.co/join](https://huggingface.co/join)
2. Accept the license at [huggingface.co/SAP/sap-rpt-1-oss](https://huggingface.co/SAP/sap-rpt-1-oss)
3. Generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Set the token:

**Windows (PowerShell):**
```powershell
$env:HUGGING_FACE_HUB_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

**Linux/Mac:**
```bash
export HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Or using .env file** (recommended):
```bash
cp .env.example .env
# Edit .env and paste your token
```

## 🧪 Quick Test

```bash
cd code
python ../scripts/test_sap_rpt1.py
```

This verifies HF token authentication, model download, and prediction accuracy.

## 📊 Run Experiments

### Single Experiment
```bash
cd code

# SAP RPT-1 OSS
python -m runners.run_experiment --dataset analcatdata_authorship --model sap-rpt1-hf

# XGBoost baseline
python -m runners.run_experiment --dataset analcatdata_authorship --model xgboost
```

### Baseline Models Only (XGBoost, CatBoost, LightGBM)
```bash
cd code

# Run on ALL datasets
python -m runners.run_baselines

# Run on specific datasets
python -m runners.run_baselines --dataset analcatdata_authorship diabetes
```

### Full Batch (All Models × All Datasets)
```bash
cd code
python -m runners.run_batch --datasets config/datasets.yaml --models config/models.yaml
```

### Available Models

| Model Name | Type | Description |
|---|---|---|
| `sap-rpt1-hf` | Pretrained (OSS) | SAP RPT-1 OSS via HuggingFace |
| `xgboost` | Baseline | XGBoost |
| `catboost` | Baseline | CatBoost |
| `lightgbm` | Baseline | LightGBM |

## 📈 View Results

Results are saved to `results/raw/[dataset]_[model].json`

Example output:
```json
{
  "dataset": "analcatdata_authorship",
  "model": "sap-rpt1-hf",
  "task_type": "classification",
  "n_samples": 841,
  "n_features": 70,
  "mean_metrics": {
    "accuracy": 1.0,
    "roc_auc": 1.0,
    "f1_macro": 1.0
  }
}
```

## 📊 Aggregate Results
```bash
cd code
python -m analysis.aggregate_results
```

## 🌐 Web Interface (New!)

We've introduced a fully interactive web application to easily benchmark your custom datasets without touching the command line.

**Features:**
- **Drag & Drop CSV Upload**: Instantly test your own data.
- **Auto-Task Detection**: Automatically detects whether your target is classification or regression.
- **Parallel Benchmarking**: Runs XGBoost, LightGBM, CatBoost, and SAP RPT-1 via cross-validation.
- **Smart Recommendations**: Evaluates models on speed, accuracy, and consistency to recommend the best option for production.

**How to start the Web App:**
```bash
cd webapp
pip install -r requirements.txt
python -m uvicorn main:app --port 8000
```
Then open your browser and navigate to `http://localhost:8000`.

## 🏗️ Project Structure

```
MINI proj SAP/
├── code/
│   ├── docker/              # Docker environments
│   ├── models/              # Model wrappers (sklearn-compatible)
│   │   ├── sap_rpt1_hf_wrapper.py  # SAP RPT-1 OSS via HuggingFace
│   │   ├── base_wrapper.py         # Abstract base class
│   │   └── ...
│   ├── datasets/            # Dataset download & preprocessing
│   ├── evaluation/          # Metrics, cross-validation, compute tracking
│   ├── runners/             # Experiment execution
│   │   ├── run_experiment.py    # Single experiment
│   │   ├── run_batch.py         # Batch experiments
│   │   └── run_baselines.py     # Baseline models only
│   ├── analysis/            # Results aggregation
│   └── config/              # YAML configurations
├── datasets/                # Downloaded dataset CSV files
├── results/                 # Experiment outputs
├── scripts/
│   └── test_sap_rpt1.py    # Quick-start validation test
├── requirements.txt         # Pinned dependencies
├── setup.py                 # Package configuration
├── docker-compose.yml       # Docker orchestration
├── .env.example             # HF token template
└── GEMINI.md                # Reproducibility checklist
```

## 🔄 Reproducibility

This repo follows NeurIPS/ICML reproducibility standards:

- **Pinned dependencies**: All packages have exact versions in `requirements.txt`
- **Fixed random seeds**: `random_state=42` across all experiments
- **Docker containers**: Isolated environments for incompatible dependencies
- **Gated model weights**: SAP RPT-1 OSS uses a fixed checkpoint (`v1.1.2`)
- **10-fold cross-validation**: Stratified splits ensure identical data partitions

See `GEMINI.md` for the full reproducibility checklist.

## 🆘 Troubleshooting

**Python version error:**
SAP RPT-1 OSS requires Python >= 3.11. Check with `python --version`.

**HF Token not working:**
```bash
huggingface-cli whoami
huggingface-cli login
```

**Docker build fails:**
```bash
docker-compose build --no-cache
```

**Out of memory:**
Edit `code/config/experiments.yaml` and reduce:
```yaml
sap_rpt1_hf:
  max_context_size: 2048  # Lower from 4096
  bagging: 1              # Lower from 4
```

