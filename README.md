# CoolSync+ — Predictive Cooling Control for AI Data Centers

## Project Description
CoolSync+ is a predictive cooling control product for AI data centers.
It uses a 4-stage AI pipeline to predict cooling demand before heat is generated.

## Pipeline
Stage 1: LLM Prompt → Token Volume
Stage 2: Token Volume → GPU Energy
Stage 3: GPU Energy → Server Room Temperature
Stage 4: Temperature → Cooling Decision

## How to Run

### Step 1: Clone the repository
git clone https://github.com/YOUR_USERNAME/coolsync-casestudy.git
cd coolsync-casestudy

### Step 2: Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter

### Step 3: Add datasets
Place the following CSV files in the data/ folder:
- stage1_processed.csv
- unified_pipeline_data.csv
- cold_source_control_dataset.csv
- final_dataset_std.csv

### Step 4: Run the notebook
jupyter notebook CoolSync_EDA.ipynb

## Datasets
| Dataset | Source | Rows |
|---------|--------|------|
| LMSYS Chatbot Arena | Kaggle | 77,792 |
| Unified Pipeline Data | Generated | 27,013 |
| Cold Source Control | Kaggle | 3,498 |
| DC Temperature Sensors | Kaggle | 27,013 |

## Team
- Aiswarya Thekkuveettil Thazhath
- Jiho Jun
- Sabrina Ronnie George Karippatt