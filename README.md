# EasyML — Autonomous AutoML & Machine Learning Studio

<p align="center">
  <b>A full-stack, no-code, and agentic machine learning platform with autonomous optimization loops, interactive explainability, and multi-file data ingestion.</b>
</p>

---

## 🌟 Key Features

### 1. 🤖 Autonomous Agentic Optimization Loop
- **PERCEIVE → PLAN → ACT → OBSERVE → REFLECT → ITERATE**
- EasyML doesn't just train one configuration; its autonomous agent acts as an in-house ML engineer. Powered by Gemini 2.5 Flash, the agent continuously diagnoses bottlenecks, overhauls preprocessing, feature selection, and model architectures, and iterates until your target metric (e.g., F1 $\ge$ 0.85 or R² $\ge$ 0.80) is achieved.
- Real-time monitor modal with live phase timeline, iteration comparison table, and streaming agent logs.

### 2. ⚡ Auto-Explore Preprocessing Engine
- Benchmarks 5 distinct preprocessing strategies (`baseline`, `standard`, `robust`, `aggressive`, `minimal`) using fast LightGBM probe cross-validation.
- Automatically discovers the optimal balance of scaling, encoding, imputation, and outlier treatment before model training.

### 3. 🔍 Explainable AI (XAI) & Interactive What-If Simulator
- **SHAP Feature Attributions**: Interactive beeswarm and bar charts revealing global and per-prediction feature importance.
- **Partial Dependence Plots (PDP)**: Marginal impact curves illustrating non-linear feature behavior.
- **Live What-If Predictor**: Sliders allow users to tweak inputs in real-time and observe immediate shifts in predicted probabilities and outcomes.

### 4. 🧠 Gemini Failure Diagnosis & Conversational Assistant
- **AI Failure Diagnosis**: When a run or step encounters issues, Gemini diagnoses the root cause in real-time and generates an actionable remediation plan.
- **Natural Language Assistant**: Floating conversational chatbot capable of loading datasets, setting targets, dispatching training, and explaining metrics.

### 5. 📂 Multi-CSV & Multi-Table Data Ingestion
- Ingest multiple CSV files, assign primary and supplementary roles, configure joins with custom keys, and exclude redundant columns with zero code.

### 6. 🛠️ Standalone Reproducible Code Generator
- One-click export generates a clean, standalone Python script (`outputs/pipeline_script.py`) mirroring the exact pipeline transformations, feature engineering, and model hyperparameters for deployment outside the studio.

---

## 🏗️ Architecture

```
Easy ML-app/
├── app.py                      # Flask REST API & Web Server
├── config.py                   # Centralized configuration & environment loader
├── database.py                 # SQLite experiment storage & run history
├── pipeline/
│   ├── agent.py                # Autonomous agentic loop (Perceive->Plan->Act->Observe->Reflect)
│   ├── ml_pipeline.py          # End-to-end ML orchestrator & Auto-Explore engine
│   ├── data_profiler.py        # Statistical dataset profiler & type inference
│   ├── data_cleaning.py        # Imputation, deduplication, outlier handling
│   ├── data_transformer.py     # Categorical encoding & numerical feature scaling
│   ├── data_splitter.py        # Stratified/Random/TimeSeries train-test splitting
│   ├── feature_engine.py       # Polynomial features, interaction terms & selection
│   ├── dim_reduction.py        # PCA, t-SNE, UMAP
│   ├── imbalance_handler.py    # SMOTE, ADASYN, Tomek, ENN, SMOTE-ENN
│   ├── model_trainer.py        # 20+ models (Classification, Regression, Clustering)
│   ├── model_evaluator.py      # K-Fold CV, hyperparameter tuning & metric evaluation
│   ├── explainer.py            # SHAP, PDP, Gemini failure diagnosis
│   ├── visualization.py        # HTML & chart report generation
│   └── code_generator.py       # Standalone pipeline_script.py exporter
├── static/
│   ├── index.html              # Glassmorphic 4-step wizard UI & Agent Monitor
│   ├── app.js                  # Client-side controller & real-time polling
│   └── style.css               # Modern dark-mode design system & animations
├── requirements.txt            # Python dependencies
└── PROJECT_STATUS.md           # Audit log & single source of truth
```

---

## 🚀 Quick Start (Native Local Setup)

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- Windows, macOS, or Linux

### 1. Clone the Repository
```bash
git clone https://github.com/siddhugitty153/EASYML.git
cd EASYML
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and insert your Gemini API Key:
```bash
cp .env.example .env
```
Edit `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
SECRET_KEY=your_secret_key_here
```

### 5. Launch EasyML
```bash
python app.py
```
Open your browser and navigate to:
```
http://localhost:5000
```

---

## 🛡️ Security & Privacy
- Sensitive configurations, API keys, and credentials are kept strictly in `.env` (which is excluded via `.gitignore`).
- Local SQLite database (`outputs/experiment_history.db`) and uploaded datasets are excluded from version control to prevent data leakage.

---

## 📄 License
This project is open-source under the MIT License.
