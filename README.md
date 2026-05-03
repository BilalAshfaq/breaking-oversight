# Breaking Oversight: Hidden Structural Signals in LLM Outputs

This project investigates whether simple hidden structural signals (e.g., acrostics) can bypass LLM-based oversight systems.

We build a controlled pipeline that generates normal and hidden text, applies multiple detection methods, and evaluates their effectiveness using semantic and structural analysis.

---

## 🚀 Overview

The pipeline consists of three main stages:

1. **Data Generation**
   - Source: CNN/DailyMail dataset
   - Generates:
     - Normal rewrite (preserves meaning)
     - Hidden rewrite (encodes secret word via sentence initials)

2. **Detection**
   - Minimal LLM detector
   - LLM rule-based detector
   - Few-shot detector
   - Deterministic rule-based detector

3. **Analysis**
   - Detection metrics (accuracy, false positives)
   - Semantic similarity (cosine similarity)
   - Visualization (PCA, t-SNE)
   - Confidence and distribution analysis

---

## 📁 Project Structure

```
.
├── dataset_gen_pipeline.py     # Data generation (normal + hidden text)
├── detection_pipeline.py       # Detection methods
├── perform_analysis.py         # Metrics + visualization
├── data_gen.sh                 # SLURM script for generation
├── detect.sh                   # SLURM script for detection + analysis
├── outputs/                    # Generated data, results, plots
└── README.md
```


---

## ⚙️ Setup

### Install dependencies
```bash
pip install transformers datasets sentence-transformers scikit-learn pandas seaborn matplotlib

```
## Model Used
Qwen/Qwen2.5-14B-Instruct

## ▶️ Running the Pipeline

### Option 1: Local execution

#### Step 1: Generate data
```bash
python dataset_gen_pipeline.py

Step 2: Run detection
python detection_pipeline.py

Step 3: Run analysis
python perform_analysis.py

```
### Option 2: SLURM (recommended for GPU)

Generate data
```bash
sbatch data_gen.sh
---

Run detection + analysis

```bash
sbatch detect.sh

```
## 📊Outputs

All results are saved in the outputs/ directory:

qwen_hidden_rewrites.csv → generated dataset

blind_detection_results.csv → detection results

analysis_with_metrics.csv → computed metrics

Plots include:

Detection performance
Semantic similarity
Word count comparison
Confidence distributions
PCA / t-SNE embeddings

## 🧪 Reproducibility
Fixed random seed used
All parameters defined in scripts
SLURM scripts automate full pipeline
Prompts used for generation and detection are provided in the report appendix

## 👤 Author

Bilal Ashfaq
