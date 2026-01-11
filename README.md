# Propensity-to-Convert Model
Performance Marketing Lead Scoring - Kissterra Home Assignment
---

# Project Overview
This project builds a machine learning pipeline to predict lead conversion probability. It compares a Logistic Regression baseline against a tuned XGBoost classifier, featuring:
* **Temporal Splitting** to prevent look-ahead bias.
* **Probability Calibration** (Isotonic) for accurate bidding.
* **SHAP Analysis** for interpretability.
* **Business Simulation** demonstrating a projected 167% increase in campaign profitability.
---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Place the Dataset
Create a `data/` folder and add the CSV file:
```
data/propensity-to-convert_dataset.csv
```

### 3. Run the Notebook
```bash
jupyter notebook propensity_model.ipynb
```

The notebook is designed to run from top to bottom (Kernel → Restart & Run All).
---

## Key Assumptions

### Prediction Moment
- **Assumption**: The model scores leads at the moment they enter the system (`created_at`), before any sales outreach occurs.
- **Rationale**: This matches the real-time bidding/ranking use case for performance marketing.

### Data Generation
- **Assumption**: Marketing features (`channel`, `bid`, `budget`, `campaign_id`) are fixed at lead creation and not retroactively updated.
- **Assumption**: Features like `crm_status`, `time_to_contact_min`, and `call_attempts` are recorded after lead intake and should not be used for prediction.

### Leakage Prevention
- **Decision**: Removed all post-conversion features (`converted_at`, `post_click_revenue`, `conversion_delay_minutes`) and process-outcome features (`crm_status`, downstream operational metrics).
- **Decision**: Used temporal + group-safe splitting to prevent future information from leaking into training.

### Outlier Handling
- **Decision**: Applied log transforms to skewed features (`bid`, `budget`) but kept all data points.
- **Rationale**: Extreme values may represent high-value "whale" campaigns; removing them would introduce selection bias.

### Model Evaluation
- **Primary metrics**: PR-AUC (for ranking under imbalance) and Log Loss (for probability quality after calibration).
- **Business metric**: Recall@10% (top-decile capture rate) to reflect operational constraints.
- **Rationale**: Propensity models are used for ranking and bidding, not hard classification.

---

## Key Results
- **Model Selection**: Tuned XGBoost outperformed the baseline in identifying high-value leads.
- **Calibration**: Reduced Log Loss from 0.60 to 0.36, ensuring reliable probability estimates.
- **Business Impact**: In a simulation of 2,000 leads, the model-guided strategy generated **$28,200 incremental profit** compared to random selection.