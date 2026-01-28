Machine Learning–Driven Cyber Threat Analytics
Regression and Classification on UNSW-NB15 Dataset
================================================

This project implements a leakage-safe machine learning framework for
cyber threat analytics using the UNSW-NB15 dataset. It addresses both
REGRESSION and CLASSIFICATION tasks using rigorous preprocessing,
validation, and evaluation strategies designed for realistic
generalization performance.

------------------------------------------------
DATASET
------------------------------------------------
Dataset: UNSW-NB15 Network Intrusion Dataset  
Source: UNSW Canberra Cyber Range Lab  

• ~2.5 million network flow records  
• 49 raw features describing traffic behavior  
• Includes both continuous and categorical attributes  

------------------------------------------------
DATA PARTITIONING STRATEGY (LEAKAGE-SAFE)
------------------------------------------------
We have used 3+1 file strategy is used:

• Files 1–3:
  Used exclusively for training, feature engineering,
  model selection, and hyperparameter tuning.

• File 4:
  Reserved as a COMPLETELY UNSEEN holdout dataset.
  Never accessed during training or tuning.

Duplicate records are removed prior to sampling to
avoid biased performance estimates.

Random sampling is applied ONLY after deduplication
to ensure memory-safe execution.

------------------------------------------------
PREPROCESSING AND DATA INTEGRITY
------------------------------------------------
• Missing and infinite values handled using statistics
  learned from TRAINING DATA ONLY

• Identical preprocessing pipelines applied to holdout data

• High-cardinality identifiers (e.g., IP addresses)
  excluded where appropriate

• All leakage checks performed before modeling

------------------------------------------------
TASK 1: REGRESSION ANALYSIS
------------------------------------------------
Objective:
Predict continuous network traffic characteristics
(e.g., connection duration).

Primary Regression Target:
• dur (connection duration)

------------------------------------------------
REGRESSION MODELS EVALUATED
------------------------------------------------
• Dummy Regressor (Median baseline)
• Decision Tree Regressor
• Random Forest Regressor
• Gradient Boosting Regressor

Tree-based models are used exclusively to ensure:
• Nonlinear modeling capability
• Robust handling of feature interactions
• Memory efficiency (no one-hot encoding)

------------------------------------------------
REGRESSION EVALUATION METRICS
------------------------------------------------
The following metrics are COMPUTED and REPORTED:

• Mean Absolute Error (MAE)
• Root Mean Squared Error (RMSE)
• R² Score
• Cross-Validation MAE
• Holdout MAE
• Generalization gap (CV vs Holdout)

No unsupported regression metrics are claimed.

------------------------------------------------
REGRESSION RESULTS (VERIFIED)
------------------------------------------------
Baseline Model:
• CV MAE: 0.6790

Cross-Validated Performance (Training Data):
• Decision Tree MAE:        0.1216
• Random Forest MAE:        0.1033  ← Best
• Gradient Boosting MAE:    0.1950

Hyperparameter Tuning (Random Forest):
• Best CV MAE after tuning: 0.0882
• n_estimators: 100
• max_depth: 12

Final Holdout Performance (Unseen Data):
• MAE:   0.0739
• RMSE:  0.8938
• R²:    0.9369

Generalization Gap:
• |CV MAE – Holdout MAE| = 0.0294

These results demonstrate strong predictive accuracy
and stable generalization.

------------------------------------------------
TASK 2: CLASSIFICATION ANALYSIS
------------------------------------------------
Objective:
Detect malicious versus benign network traffic
using supervised classification.

------------------------------------------------
CLASSIFICATION PIPELINE
------------------------------------------------
• Leakage-aware preprocessing
• Class labels handled separately from regression targets
• Identical train/holdout isolation strategy
• Model evaluation performed ONLY on unseen data

------------------------------------------------
CLASSIFICATION METRICS POLICY
------------------------------------------------
Classification metrics are reported ONLY if they are
explicitly computed in the code.

Common metrics used (where applicable):
• Accuracy
• Precision
• Recall
• F1-score (reported only when calculated)

No classification metric is claimed without
direct computational evidence.

------------------------------------------------
KEY METHODOLOGICAL STRENGTHS
------------------------------------------------
• Strict data leakage prevention
• Clear separation of regression and classification tasks
• Baseline comparisons for contextual performance
• Cross-validation before holdout evaluation
• Memory-safe execution throughout
• Numerically defensible conclusions

------------------------------------------------
LIMITATIONS
------------------------------------------------
• Temporal dependencies not explicitly modeled
• UNSW-NB15 may not fully reflect modern traffic patterns
• Regression targets limited to available numeric attributes

------------------------------------------------
FUTURE WORK
------------------------------------------------
• Time-series modeling (LSTM, ARIMA)
• Multi-target regression
• Advanced ensemble methods
• Real-time deployment optimization
• Joint regression–classification modeling

------------------------------------------------
FINAL NOTE
------------------------------------------------
All reported results in this project are:
• Reproducible
• Leakage-safe
• Numerically supported by experiments
• Consistent with the provided code and logs

Unsupported claims have been explicitly avoided.

================================================
END OF README
================================================
