# CoolSync+ — Predictive Cooling Control System for AI Data Centers

### CSCN8040 — Case Study in AIML & Data Science | Team 4
**Conestoga College | Winter 2026**

---

## What is CoolSync+?

**CoolSync+** is a predictive cooling control product designed for GPU-dense AI data centers.

Traditional cooling systems react **after** temperatures rise — creating dangerous thermal spikes,
wasting energy, and degrading GPU performance. CoolSync+ solves this by **predicting cooling
demand before heat is generated**, using a four-stage AI pipeline that connects LLM prompt
activity directly to proactive cooling decisions.

> Current data centers spend **45–50%** of total energy on cooling.
> PID-based reactive systems create a **3–8 minute response lag** after temperature spikes.
> CoolSync+ targets **≥30% cooling energy reduction** and a PUE improvement from **1.56 → 1.30**.

---

## The Four-Stage Product Pipeline

```
LLM Prompt
    ↓
Stage 1: Prompt → Token Volume
    ↓
Stage 2: Token Volume → GPU Energy
    ↓
Stage 3: GPU Energy → Server Room Temperature
    ↓
Stage 4: Temperature + Workload → Cooling Decision
```

---

## Repository Structure

```
CSCN8040_CoolSync_CaseStudy/
│
├── data/
│   ├── stage1_token_prediction/
│   │   └── stage1_processed.csv              (77,792 LLM conversations)
│   ├── stage3_temperature_prediction/
│   │   └── final_dataset_std.csv             (27,013 sensor readings)
│   ├── stage4_cooling_control/
│   │   └── cold_source_control_dataset.csv   (3,498 cooling records)
│   └── unified_pipeline_data.csv             (27,013 pipeline observations)
│
├── results/
│   └── figures/
│       ├── figure1_stage1_token_analysis.png
│       ├── figure2_stage2_token_analysis.png
│       ├── figure3_stage3_temperature.png
│       ├── figure4_stage4_cooling_analysis.png
│       ├── figure5_hypothesis_test.png
│       └── figure6_ml_model.png
│
├── CoolSync_CaseStudy.ipynb                  (main notebook — run this)
├── README.md
└── .gitignore
```

---

## How to Run

### Requirements

- Python 3.9 or higher
- Jupyter Notebook or VS Code with Jupyter extension

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/AiswaryaSukumar/CSCN8040_CoolSync_CaseStudy.git
cd CSCN8040_CoolSync_CaseStudy
```

---

### Step 2 — Install Dependencies

Run this single command to install all required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
```

| Package | Purpose |
|---------|---------|
| pandas | Data loading, cleaning, and manipulation |
| numpy | Numerical calculations and array operations |
| matplotlib | All charts and visualizations |
| seaborn | Styled heatmaps and statistical plots |
| scikit-learn | Random Forest model training and evaluation |
| scipy | Welch T-Test for hypothesis validation |
| jupyter | Notebook execution environment |

---

### Step 3 — Verify Dataset Availability

Confirm the following files exist in your local `data/` folder:

```
data/stage1_token_prediction/stage1_processed.csv
data/stage3_temperature_prediction/final_dataset_std.csv
data/stage4_cooling_control/cold_source_control_dataset.csv
data/unified_pipeline_data.csv
```

> **Note:** The raw file `chatbot_arena_conversations.json` (108MB) is excluded
> from this repository due to GitHub's 100MB file size limit.
> The notebook uses the pre-processed file `stage1_processed.csv` which is included.
> The original raw file can be downloaded from:
> https://www.kaggle.com/datasets/lmsysorg/chatbot-arena-conversations

---

### Step 4 — Run the Notebook

```bash
jupyter notebook CoolSync_CaseStudy.ipynb
```

Or open in VS Code and select **Kernel → Restart and Run All**.

> All 6 figures will be generated automatically and saved to `results/figures/`.
> No additional configuration is required.
> Run all cells from top to bottom in order.

---

## Datasets

| # | Dataset | Source | Rows | Columns | Role |
|---|---------|--------|------|---------|------|
| 1 | LMSYS Chatbot Arena Conversations | [Kaggle](https://www.kaggle.com/datasets/lmsysorg/chatbot-arena-conversations) | 77,792 | 7 | Stage 1 — LLM workload proxy |
| 2 | Unified Pipeline Data | Generated | 27,013 | 7 | Stage 2/3 — token to power pipeline |
| 3 | DC Temperature Sensor Data | [Kaggle](https://www.kaggle.com/datasets/mbjunior/data-centre-hot-corridor-temperature-prediction) | 27,013 | 43 | Stage 3 — thermal load indicators |
| 4 | Cold Source Control Dataset | [Kaggle](https://www.kaggle.com/datasets/programmer3/data-center-cold-source-control-dataset) | 3,498 | 12 | Stage 4 — cooling energy and actions |

---

## Notebook Structure

| Section | Description |
|---------|-------------|
| Section 1 | Environment setup and package installation |
| Section 2 | Data loading and quality verification |
| Section 3 | Data preparation and orchestration |
| Section 4 | Exploratory Data Analysis — 4 connected stages |
| Section 5 | Statistical Hypothesis Testing — Welch T-Test (A/B Experiment) |
| Section 6 | Machine Learning Model — Random Forest Regressor (AIML Tool) |
| Section 7 | CoolSync+ Product Description (markdown) |
| Section 8 | Conclusions, Next Steps, and References |

---

## Research Hypothesis

| | Statement |
|--|-----------|
| **H₀ (Null)** | Server workload level does NOT significantly affect cooling energy consumption |
| **H₁ (Alternative)** | Higher server workload requires significantly more cooling energy consumption |

**Independent Variable (IV):** Server workload intensity (%)
**Dependent Variable (DV):** Cooling unit power consumption (kW)
**Test:** Independent samples Welch T-Test
**Significance level:** α = 0.05

---

## Key Results

| Metric | Result | Status |
|--------|--------|--------|
| Stage 2: Peak vs off-peak requests | +133.8% during business hours | Predictable pattern confirmed |
| Stage 3: T_out sensor correlation with TLHC | r = 0.512 | Moderate-strong correlation |
| Stage 4: Workload vs cooling correlation | r = 0.935 | Very strong correlation |
| T-Test T-statistic | 72.80 | Highly significant |
| T-Test p-value | < 0.0001 | H₀ rejected |
| T-Test Cohen's d | 2.46 | Large practical effect |
| Energy difference (high vs low workload) | +38.8% | H₁ supported |
| Random Forest R² | 0.8989 | PASSED (target > 0.85) |
| Random Forest MAE | 0.0427 kW | PASSED (target < 0.05) |
| Server_Workload feature importance | 0.9214 (92%) | Primary cooling driver confirmed |

---

## AIML Tool — Random Forest Regressor

The Random Forest Regressor (Section 6 of the notebook) is the AIML tool
trained and executed in this case study.

- **Input features:** Server workload, inlet temperature, ambient temperature, hour of day, peak hour flag
- **Target:** Cooling unit power consumption (kW)
- **Training split:** 80% train / 20% test (random state = 42)
- **Estimators:** 100 trees, max depth = 10
- **R² = 0.8989** — exceeds user acceptance threshold of 0.85
- **MAE = 0.0427 kW** — within user acceptance threshold of 0.05 kW
- **Zero manual configuration required**

---

## Source Evaluation — CRAAP Method

All sources cited in this case study were evaluated using the CRAAP method:

| Component | Assessment |
|-----------|-----------|
| **Currency** | All primary references published 2024–2026, within 24 months of submission |
| **Relevance** | Each source directly addresses predictive cooling, LLM workloads, or RL for data centers |
| **Authority** | Peer-reviewed papers from IEEE Transactions, ACM ASPLOS, and Applied System Innovation |
| **Accuracy** | Claims cross-validated across multiple independent papers and real dataset measurements |
| **Purpose** | All sources cited for scientific support, not commercial promotion |

---

## Team



| Aiswarya Thekkuveettil Thazhath 

| Jiho Jun 

| Sabrina Ronnie George Karippatt 

---

## References

1. Stojkovic, J. et al. (2025). TAPAS: Thermal and power-aware scheduling for LLM inference. ACM ASPLOS Vol. 2, 1266–1281.
2. Liu, J. et al. (2026). Proactive Cooling Control Algorithm Based on LSTM-Driven Predictive Thermal Analysis. Applied System Innovation, 9(1), 21. https://doi.org/10.3390/asi9010021
3. Abera, N. B. & Chen, Y. (2026). Coordinated Cooling and Compute Management for AI Datacenters. IEEE Transactions on Cloud Computing. arXiv:2601.08113v1
4. Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324
5. Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.). Lawrence Erlbaum Associates.

---

*CoolSync+ | CSCN8040 Case Study | Conestoga College | Winter 2026*
