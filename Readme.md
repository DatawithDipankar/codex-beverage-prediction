# 🥤 CodeX Beverage: Price Prediction

> A machine learning solution to predict the preferred price range of energy drink customers  built for CodeX Beverage as part of a data science internship project.

---

##  Project Overview

CodeX Beverage wanted to understand what price range their customers prefer when purchasing energy drinks. Using a survey dataset of **30,000+ respondents** across India, this project builds a complete ML pipeline from raw data cleaning to a deployed Streamlit web application that predicts a customer's preferred price range based on their demographic and behavioral profile.

---

##  Problem Statement

Given a customer's profile (age, zone, income, consumption habits, brand preferences, etc.), predict which price range they are most likely to purchase from:

| Class | Price Range |
|-------|------------|
| 0 | ₹50 - ₹100 |
| 1 | ₹100 - ₹150 |
| 2 | ₹150 - ₹200 |
| 3 | ₹200 - ₹250 |

---

##  Project Structure

```
codex-beverage-prediction/
│
├── app.py                        # Streamlit web application
├── requirements.txt              # Python dependencies
├── lgbm_model.pkl                # Trained LightGBM model
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb    # Data cleaning pipeline
│   ├── 02_feature_engineering.ipynb  # Feature creation
│   └── 03_modeling.ipynb         # Model training & evaluation
│
├── data/
│   └── survey_results.csv        # Raw survey dataset
│
└── README.md
```

---

##  Project Pipeline

```
Raw Data (30,010 rows)
        ↓
Step 1: Remove Duplicates          → 30,000 rows
        ↓
Step 2: Remove Invalid Ages        → 29,991 rows  (age > 100 removed)
        ↓
Step 3: Handle Missing Values      → 29,991 rows
        ↓
Step 4: Fix Spelling Mistakes      → 29,991 rows
        ↓
Step 5: Feature Engineering        → 29,956 rows  (logical outliers removed)
        ↓
Step 6: Encoding + Model Training
        ↓
Step 7: Streamlit Deployment
```

---

##  Data Cleaning

| Step | Action | Result |
|------|--------|--------|
| Remove Duplicates | Drop duplicate rows | 30,010 → 30,000 |
| Outlier Detection | Remove ages > 100 (invalid entries like 192, 604) | 30,000 → 29,991 |
| Missing Values | `income_levels` → "Not Reported", `consume_frequency` & `purchase_channel` → mode | 0 nulls remaining |
| Spelling Fixes | Fixed `Metor→Metro`, `urbna→Urban`, `Establishd→Established`, `newcomer→Newcomer` | Uniform categories |
| Logical Outliers | Remove students in 56-70 age group (illogical entries) | 29,991 → 29,956 |

---

##  Feature Engineering

Three new features were created to enrich the dataset:

### 1. `age_group`
Age was binned into groups: `18-25`, `26-35`, `36-45`, `46-55`, `56-70`, `70+`

### 2. `cf_ab_score` - Consume Frequency & Awareness Brand Score
Captures how likely a customer is to be engaged with the market:
```
cf_ab_score = frequency_score / (awareness_score + frequency_score)
```

### 3. `zas_score` - Zone Affluence Score
Combines geographic zone and income level to estimate purchasing power:
```
zas_score = zone_score × income_score
```

### 4. `bsi` - Brand Switching Indicator
Binary flag (0/1) indicating if a customer is likely to switch brands:
```
bsi = 1 if (current_brand ≠ 'Established') AND (reason IN ['Price', 'Quality'])
```

---

##  Model Training

**Train/Test Split:** 75% training | 25% testing | `random_state=42`

**Encoding:**
- Label Encoding: `age_group`, `income_levels`, `health_concerns`, `consume_frequency(weekly)`, `preferable_consumption_size`
- One-Hot Encoding: all remaining categorical columns
- Label Encoding: target variable `price_range`

###  Model Comparison

| Model | Accuracy |
|-------|----------|
| 🥇 **LightGBM** | **92.16%** |
| 🥈 XGBoost | 92.00% |
| 🥉 Random Forest | 89.09% |
| SVM | 82.45% |
| Logistic Regression | 80.08% |
| Gaussian Naive Bayes | 58.31% |

**Best Model: LightGBM** with **92.16% accuracy**  selected for deployment.

---

##  Streamlit App

The app takes customer inputs across 4 sections and predicts their preferred price range in real time.

**Input Sections:**
-  Demographics (Age, Gender, Zone, Occupation)
-  Purchase Behaviour (Income, Frequency, Brand, Size)
-  Preferences & Awareness (Awareness, Reasons, Flavor, Channel)
-  Packaging & Lifestyle (Packaging, Health Concerns, Consumption Situation)

**How to run locally:**
```bash
streamlit run app.py
```

---

##  Deployment

This app is deployed on **Streamlit Cloud**.

👉 **Live App:** [your-app-link.streamlit.app](https://your-app-link.streamlit.app)

---

##  Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost, LightGBM |
| Experiment Tracking | MLFlow, DagsHub |
| Web App | Streamlit |
| Version Control | Git, GitHub |

---

##  Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/codex-beverage-prediction.git
cd codex-beverage-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

##  Results

The LightGBM model achieves **92.16% accuracy** with strong performance across all 4 price range classes:

| Price Range | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| ₹50-100 | 0.92 | 0.91 | 0.92 |
| ₹100-150 | 0.90 | 0.90 | 0.90 |
| ₹150-200 | 0.90 | 0.91 | 0.90 |
| ₹200-250 | 0.96 | 0.95 | 0.96 |

---

## 👤 Author

**Dipankar**
- GitHub: [@DatawithDipankar](https://github.com/DatawithDipankar)
- LinkedIn: [linkedin](https://linkedin.com/in/dipankar-mane/)

---

## 📄 License

This project is part of a Codebasics data science internship assignment.