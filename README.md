# Machine Learning-Driven Cyber Threat Analytics
## Regression and Classification on UNSW-NB15 Dataset

This project implements a comprehensive,  machine learning framework for cyber threat analytics using the UNSW-NB15 dataset. It addresses both binary/multi-class classification and regression tasks using rigorous preprocessing, feature engineering,leakage-safe, duplicate removal, and validation strategies designed for realistic generalization performance.

**Submitted By:**
- Ashutosh Kumar Verma (Student ID: 475852)
- Ramik Sharma (Student ID: 477656)

---

## Dataset

**Dataset:** UNSW-NB15 Network Intrusion Detection Dataset  
**Source:** University of New South Wales (UNSW), Canberra Cyber Range Lab

- Approximately 2.5 million network flow records distributed across 4 CSV files
- 49 raw features capturing network flow statistics, protocol information, and behavioral attributes
- Includes both continuous and categorical attributes
- Contains labeled data for both classification and regression tasks

---

## Data Partitioning Strategy

We have implemented a strict 3+1 file strategy to ensure unbiased evaluation:

### Training Data (Files 1–3)
Used exclusively for:
- Exploratory data analysis
- Feature engineering
- Model training
- Hyperparameter tuning
- Cross-validation

### Holdout Data (File 4)
- Completely unseen validation dataset
- Never accessed during training, feature selection, or tuning
- Used only for final model evaluation
- Provides realistic deployment performance estimates

### Data Quality Measures
- **Duplicate removal:** Approximately 22% duplicate rate detected and removed from training data
- **Cross-file overlap check:** 60 overlapping records identified between training and holdout
- **Stratified sampling:** 50,000 training samples from combined files 1 to 3  and 10,000 holdout samples from file 4 used as holdout data for computational efficiency
- All preprocessing parameters learned from training data only

---

## Preprocessing and Data Integrity

### Comprehensive Data Cleaning
- **Missing value handling:** Median imputation for numeric features, mode for categorical
- **Infinite value replacement:** Converted to NaN and imputed
- **Duplicate detection:** Systematic analysis before and after file merging
- **Leakage prevention:** Features flagged and removed through semantic analysis

### Feature Engineering
- **Byte-based features:** total_bytes, byte_ratio
- **Packet-based features:** total_packets, packet_ratio
- **TTL features:** ttl_diff, ttl_ratio, ttl_sum
- **Load features:** load_ratio
- **Window features:** window_ratio, window_diff
- **Loss features:** loss_ratio, total_loss
- **Log-transformed ports:** Reduced skewness in port number distributions

### Leakage Analysis
Systematic screening identified and removed:
- **Temporal features:** Stime, Ltime (timestamps)
- **Attack category:** attack_cat (direct label information)
- **High-correlation features:** Manual review of features with absolute correlation > 0.95

---

## Task 1: Binary Classification

### Objective
Detect malicious versus benign network traffic with emphasis on recall optimization (minimizing missed attacks).

### Target Variable
- **Label:** Binary indicator (0 = Normal, 1 = Attack)
- **Class distribution:** Approximately 95% benign, 5% malicious (highly imbalanced)

### Classification Models Evaluated
- Dummy Classifier (most frequent baseline)
- Logistic Regression (class-weighted)
- Decision Tree (class-weighted)
- Random Forest (class-weighted)
- Gradient Boosting
- Naive Bayes

All models use class_weight='balanced' where applicable to handle class imbalance.

### Feature Selection
- **Method:** SelectKBest with f_classif (ANOVA F-statistic)
- **Selected features:** Top 20 features for binary classification
- **Key features:** sttl, ct_state_ttl, dttl, tcprtt, synack, ackdat

### Classification Results

#### Baseline Performance
- **Accuracy:** 86% (predicting all samples as benign)
- **Recall:** 0% (no attacks detected)

#### Cross-Validated Performance (5-fold CV)

| Model                  | CV Recall (Mean ± Std) | Accuracy | AUC    |
|------------------------|------------------------|----------|--------|
| Logistic Regression    | 0.9976 ± 0.0017        | 0.9850   | 0.9962 |
| Random Forest          | 0.9934 ± 0.0107        | 0.9910   | 0.9998 |
| Gradient Boosting      | 0.9819 ± 0.0039        | 0.9954   | 0.9998 |
| Decision Tree          | 0.9808 ± 0.0218        | 0.9908   | 0.9981 |
| Naive Bayes            | 0.9458 ± 0.0162        | 0.9784   | 0.9963 |

**Best Model:** Logistic Regression (selected for highest recall)

#### Test Set Performance (Best Model)

```
              precision    recall  f1-score   support
     Normal       1.00      0.98      0.99      8620
     Attack       0.90      1.00      0.95      1380

   accuracy                           0.98     10000
```

- **Test Accuracy:** 98.5%
- **Test AUC:** 0.9962
- **Confusion Matrix:** 3 false negatives, 147 false positives

#### Holdout Validation (Unseen Data)

```
              precision    recall  f1-score   support
     Normal       1.00      0.79      0.88      9289
     Attack       0.27      1.00      0.42       710

   accuracy                           0.80      9999
```

- **Holdout Accuracy:** 80.33%
- **Holdout AUC:** 0.9830
- **Holdout Recall:** 100% (all attacks detected)
- **Confusion Matrix:** 0 false negatives, 1967 false positives

### Model Selection Rationale
Logistic Regression was selected as the final model based on:
1. Highest cross-validated recall (0.9976)
2. Perfect recall on holdout data (100%)
3. Model interpretability for security applications
4. Stable generalization across datasets

While tree-based models achieved slightly higher AUC, the recall-first approach aligns with cybersecurity priorities where missing attacks (false negatives) is more costly than false alarms.

### Sanity Check: Label Shuffling
To verify absence of data leakage:
- **Shuffled labels training:** AUC dropped to 0.6045 (near-random)
- **Duplicate rate verification:** 0.14% after preprocessing
- **Conclusion:** Model learns genuine patterns, not artifacts

---

## Task 2: Regression Analysis

### Objective
Predict continuous network traffic characteristics to support capacity planning, performance monitoring, and anomaly detection through residual analysis.

### Regression Targets
Eight continuous-valued network attributes:
- **dur** – Connection duration (seconds) - Primary target
- **sbytes** – Source to destination bytes
- **dbytes** – Destination to source bytes
- **Sload** – Source bits per second
- **Dload** – Destination bits per second
- **tcprtt** – TCP round-trip time
- **synack** – TCP SYN-ACK response time
- **ackdat** – TCP ACK-DATA transmission time

### Regression Models Evaluated
Tree-based models exclusively (memory-efficient, no scaling required):
- Dummy Regressor (median baseline)
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

### Feature Engineering (Regression-Specific)
Additional features beyond classification:
- **Packet ratios:** packet_ratio
- **Window metrics:** window_ratio, window_diff
- **TTL aggregations:** ttl_sum, ttl_ratio
- **Loss metrics:** loss_ratio, total_loss
- **Log-transformed ports:** sport_log, dsport_log

### Regression Results (Target: dur)

#### Baseline Performance
- **CV MAE:** 0.6790 (median predictor)

#### Cross-Validated Performance (3-fold CV)

| Model              | CV MAE (Mean ± Std) |
|--------------------|---------------------|
| Random Forest      | 0.1033 ± 0.0065     |
| Decision Tree      | 0.1216 ± 0.0022     |
| Gradient Boosting  | 0.1950 ± 0.0067     |

All models substantially outperform baseline (6.5x improvement).

#### Hyperparameter Tuning (Random Forest)
Optimal configuration via GridSearchCV:
- **n_estimators:** 100
- **max_depth:** 12
- **min_samples_split:** 2
- **min_samples_leaf:** 1
- **Best CV MAE:** 0.0882

#### Final Holdout Performance (Unseen Data)
- **Holdout MAE:** 0.0739
- **Holdout RMSE:** 0.8938
- **Holdout R²:** 0.9369 (93.7% variance explained)

#### Generalization Gap Analysis
- **CV MAE (best model):** 0.1033
- **Holdout MAE:** 0.0739
- **Absolute difference:** 0.0294
- **Interpretation:** Strong generalization, minimal overfitting

### Feature Importance (Top 15 for dur)
1. Dintpkt (0.468)
2. Sintpkt (0.220)
3. total_packets (0.111)
4. Spkts (0.081)
5. Sjit (0.043)
6. packet_ratio (0.034)
7. total_loss (0.033)
8. dloss (0.008)
9. Dpkts (0.001)
10. Djit (0.001)

---

## Evaluation Metrics Summary

### Classification Metrics (Computed)
- Accuracy
- Precision
- Recall (primary optimization metric)
- F1-score
- AUC-ROC
- Confusion Matrix
- Cross-validation scores

### Regression Metrics (Computed)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score (Coefficient of Determination)
- Cross-validation MAE
- Holdout MAE
- Generalization gap

---

## Key Summary

### Data Quality
- Systematic duplicate detection and removal (22% training duplicates)
- Cross-dataset overlap analysis (60 overlapping records identified)
- Missing value analysis and robust imputation
- Infinite value handling

### Leakage Prevention
- Strict 3+1 file partitioning strategy
- Temporal feature removal (Stime, Ltime)
- Feature engineering performed identically on train/holdout
- All preprocessing parameters learned from training data only
- No feature selection or scaling on holdout data

### Model Validation
- Baseline comparisons establish performance floor
- Cross-validation before holdout evaluation
- Hyperparameter tuning on training data only
- Final evaluation on completely unseen data
- Label shuffling sanity check (classification)

### Computational Efficiency
- Memory-safe streaming data loading
- Stratified sampling for feasibility
- Tree-based models (no one-hot encoding explosion)
- Single-threaded execution to prevent memory spikes
---

## Repository Structure

```
cyber-threat-analytics/
├── README.md                                    # This file
├── cyber_threat_classification.pdf     # Classification report
├── cyber-threat-regression.pdf         # Regression report
├── notebooks/
│   ├── 01_classification_analysis.ipynb         # Binary classification
│   └── 02_regression_analysis.ipynb             # Regression modeling
├── data/
│   ├── UNSW-NB15_1.csv                          # Training file 1
│   ├── UNSW-NB15_2.csv                          # Training file 2
│   ├── UNSW-NB15_3.csv                          # Training file 3
│   ├── UNSW-NB15_4.csv                          # Holdout file
│   └── NUSW-NB15_features.csv                   # Feature metadata

---

**Last Updated:** January 2026
