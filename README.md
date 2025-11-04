# IoT Device Network Audit — Binary & Multiclass Intrusion Detection System (IDS)

**Author:** Oleksandr Kuznetsov  
**Affiliation:** Università eCampus, Italy  
**License:** MIT  
**Version:** 0.1.0  

---

## 🔍 Overview

This repository provides an **end-to-end machine learning pipeline** for auditing IoT and IIoT network traffic security.  
It supports both **binary** (attack vs normal) and **multiclass** (attack type) classification tasks.

The system performs:
- structured preprocessing and feature encoding;
- exploratory data analysis and visualization;
- supervised training using modern ensemble models;
- quantitative comparison across accuracy, F1, ROC/PR AUC, and inference latency;
- per-class analysis for model explainability and reliability auditing.

---

## 🧠 Research Context

IoT devices are among the most vulnerable elements of modern digital ecosystems.  
This project focuses on **auditable and interpretable IDS models** that can:
- detect common attack patterns in network flows (DoS, DDoS, MITM, password, ransomware, etc.),
- evaluate **robustness, latency, and resource footprint**, and  
- provide a baseline for **TinyML and federated IDS** deployments in smart environments.

The work is aligned with ongoing EU research directions in **edge security, federated analytics, and trustworthy AI**.

---

## ⚙️ Architecture Overview
```
src/
└── iot_audit/
├── preprocessing.py / preprocessing_mc.py # Data loading & encoding (binary / multiclass)
├── metrics.py / metrics_mc.py # Metrics, visualization, reports
scripts/
├── analyze_dataset.py, visualize_dataset.py # EDA, summary statistics, correlation plots
├── train_.py # Model training (RF, LGBM, XGB, LogReg)
├── compare_models.py # Binary comparison
├── train_mc_.py # Multiclass variants
└── compare_models_mc.py # Multiclass comparison and benchmarks
reports*/ # Generated metrics, plots, and summaries
```

Each model is isolated under its own folder, ensuring reproducibility and traceability.

---

## 📊 Benchmark Summary (snapshot)

| Model     | Accuracy | Macro-F1 | ROC-AUC Micro | Total Size (MB) | Inference (ms/1k) |
|------------|-----------|-----------|---------------|-----------------|-------------------|
| **LGBM-MC** | 0.9903    | 0.9694    | 0.99994       | 29.3            | 45.39             |
| RF-MC      | 0.9897    | 0.9681    | 0.99989       | 46.2            | 31.10             |
| XGB-MC     | 0.9889    | 0.9665    | 0.99988       | 40.8            | 52.23             |
| LogReg-MC  | 0.8122    | 0.7830    | 0.9213        | 0.019           | **5.44**          |

> All models were trained on the same dataset (`train_test_network.csv`, ~211k flows, 44 columns).  
> Metrics: stratified 80/20 split, consistent seed = 42.

---

## Highlights
- Clean project layout: `src/`, `scripts/`, `reports*/` (artifacts), `figures/`.
- Reproducible preprocessing (imputation + one‑hot).
- Models: RandomForest, LightGBM, XGBoost, Logistic Regression (binary & multiclass).
- Metrics: accuracy, F1, ROC‑AUC/PR‑AUC, confusion matrix, per‑class report.
- Comparison scripts (quality, size, latency); artifacts per model in isolated folders.
- Ready for GitHub CI: lint + basic import smoke.

## Dataset
Place your CSV (e.g., `data/train_test_network.csv`) with columns like:
```
src_ip,src_port,dst_ip,dst_port,proto,service,duration,src_bytes,dst_bytes,conn_state,...,label,type
```
- `label` → binary target (0=normal, 1=attack)
- `type`  → multiclass target (e.g., normal, ddos, dos, scanning, injection, xss, ransomware, password, backdoor, mitm)

> Note: heavy free‑text columns (e.g., `http_uri`, `ssl_subject`) are dropped by default.

## Quickstart

```bash
# 1) Create env
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell
# source .venv/bin/activate  # Linux/Mac

# 2) Install deps
pip install -U pip
pip install -r requirements.txt

# 3) Put data
# data/train_test_network.csv

# 4) EDA
python scripts/analyze_dataset.py --csv data/train_test_network.csv --outdir reports
python scripts/visualize_dataset.py --csv data/train_test_network.csv --outdir reports/figures

# 5) Binary training
python scripts/train_rf.py    --csv data/train_test_network.csv --outdir reports
python scripts/train_lgbm.py  --csv data/train_test_network.csv --outdir reports
python scripts/train_xgb.py   --csv data/train_test_network.csv --outdir reports
python scripts/train_logreg.py --csv data/train_test_network.csv --outdir reports

# 6) Binary comparison
python scripts/compare_models.py --outdir reports --models rf lgbm xgb logreg --benchmark --sample_size 10000

# 7) Multiclass training
python scripts/train_mc_rf.py    --csv data/train_test_network.csv --outdir reports_mc
python scripts/train_mc_lgbm.py  --csv data/train_test_network.csv --outdir reports_mc
python scripts/train_mc_xgb.py   --csv data/train_test_network.csv --outdir reports_mc
python scripts/train_mc_logreg.py --csv data/train_test_network.csv --outdir reports_mc

# 8) Multiclass comparison
python scripts/compare_models_mc.py --outdir reports_mc --models rf_mc lgbm_mc xgb_mc logreg_mc --benchmark --sample_size 10000
```

## Artifact layout

```
reports/
  models/
    rf|lgbm|xgb|logreg/
      model.pkl
      preprocessor.pkl
      metrics.json
      leakage_report.json
      feature_importances.csv
  figures/
    rf|lgbm|xgb|logreg/
      roc_curve.png, pr_curve.png, confusion_matrix.png, feature_importances_top30.png

reports_mc/
  models/
    <model>_mc/
      model.pkl, preprocessor.pkl, metrics.json, per_class_report.csv, label_map.json, feature_importances.csv
  figures/
    <model>_mc/
      confusion_matrix.png, pr_micro.png, pr_<k>_<class>.png, feature_importances_top30.png
  summary/
    summary_models_mc.csv, per_class_report_merged.csv, accuracy.png, macro_f1.png, ...
```

## Reproducibility & Notes
- Stratified split (80/20).
- Known leakage columns are dropped (e.g., `type` in binary task).
- Probabilities used to compute ROC/PR curves; safe handling if unavailable.
- For fair comparison, use the same CSV and seeds.

## Results (example snapshot)
- **LGBM‑MC**: accuracy ~0.9903, macro‑F1 ~0.9694, ROC‑AUC micro ~0.99994.
- **RF‑MC**: accuracy ~0.9897, macro‑F1 ~0.9681 (close to LGBM‑MC).
- Inference latency per 1k flows (10k sample): `logreg_mc` ~**5.44ms**, `lgbm_mc` ~**45.39ms**.

> Adjust thresholds for risk appetite: minimize FP for production or maximize recall on critical classes.

## 📚 Dataset: TON_IoT Network Dataset

**Name:** TON_IoT Network Dataset — IoT/IIoT network traffic for intrusion detection  
**Provider:** Cyber Range & IoT Labs, UNSW Canberra (SEIT) — *TON_IoT dataset collection*  
**Official page:** https://research.unsw.edu.au/projects/toniot-datasets  
**License:** Creative Commons **Attribution 4.0 International (CC BY 4.0)** (see the TON_IoT site for details)

This repository uses the *train/test* network flows subset often distributed as
`train_test_network.csv` (~29.9 MB; 44 columns) (https://www.kaggle.com/datasets/arnobbhowmik/ton-iot-network-dataset). 
The flows were captured in realistic
IoT/IIoT smart-environment scenarios using tools such as **Argus** and **Bro (Zeek)**.
The dataset contains **benign** and **malicious** traffic and is suitable for
intrusion detection, anomaly detection, and ML benchmarking.

> **Columns (examples, 10 of 44):**
> `src_ip, src_port, dst_ip, dst_port, proto, service, duration, src_bytes, dst_bytes, conn_state, ...`
>
> Targets used in this repo:
> - `label` — binary (0 = normal, 1 = attack)  
> - `type`  — multiclass (e.g., `normal, ddos, dos, scanning, injection, xss, ransomware, password, backdoor, mitm`)

**Notes & caveats**
- Some high-cardinality text fields (e.g., `http_uri`, `ssl_subject`, `ssl_issuer`) are dropped by default to avoid leakage and reduce sparsity.
- Distribution is imbalanced across classes (e.g., `mitm` is rare). We report **macro-F1** and per-class metrics.
- Source of the CSV used here: community mirror (e.g., Kaggle: *ToN_IoT Network Dataset*). Refer to the **official UNSW page** for canonical downloads and documentation.

### 📝 Acknowledgments
We gratefully acknowledge **The TON_IoT Datasets** team at **UNSW Canberra** for creating and maintaining the dataset collection. Free academic use is permitted under CC BY 4.0; for commercial use consult the dataset authors.

### 📖 Recommended citations (as provided by TON_IoT)
- Moustafa, N. “A new distributed architecture for evaluating AI-based security systems at the edge: Network TON_IoT datasets.” *Sustainable Cities and Society* (2021): 102994.  
- Booij, T. M., Chiscop, I., Meeuwissen, E., Moustafa, N., den Hartog, F. T. H. “ToN IoT—The role of heterogeneity… in IoT network intrusion datasets.” *IEEE IoT Journal* (2021).  
- Alsaedi, A., Moustafa, N., Tari, Z., Mahmood, A., Anwar, A. “TON_IoT telemetry dataset…” *IEEE Access* 8 (2020): 165130-165150.  
- Moustafa, N., Keshk, M., Debie, E., Janicke, H. “Federated TON_IoT Windows Datasets…” *IEEE TrustCom* (2020): 848-855.  
- Moustafa, N., Ahmed, M., Ahmed, S. “Data Analytics-Enabled Intrusion Detection: Evaluations of ToN_IoT Linux Datasets.” *IEEE TrustCom* (2020): 727-735.  
- Moustafa, N. “New Generations of IoT Datasets for Cybersecurity Applications based ML: TON_IoT Datasets.” *eResearch Australasia* (2019).

> Please cite the relevant TON_IoT papers when publishing results that use this dataset.

## How to contribute
See [CONTRIBUTING.md](CONTRIBUTING.md). Please run lint before commits.

## Citation
See [CITATION.cff](CITATION.cff).

## License
[MIT](LICENSE)