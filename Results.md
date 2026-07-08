# Results — Space Mission Risk Analytics

Held-out test set metrics for the final model. All metrics are computed on a **chronological split** (most recent ~20% of launches), not a random split. The model never sees future data relative to what it is evaluated on.

---

## Final Model: Calibrated Random Forest

| Metric | Uncalibrated | Calibrated (Isotonic) |
| :--- | :---: | :---: |
| ROC-AUC | 0.7489 | 0.7489 |
| Brier Score | 0.1648 | 0.0472 |
| PR-AUC (Failure Class) | 0.3400 | 0.3306 |
| Mean Calibration Error | — | 0.0370 |

**Test Set Composition:** 840 launches, including 48 failures (5.7% failure rate).

> **Note:** The failure rate differs from the full dataset's overall success rate (90.4%) because evaluation is performed on a chronological split containing the most recent launches.

### Why the ROC-AUC Remains Unchanged After Calibration

Probability calibration is intended to improve the reliability of predicted probabilities without changing the model's ranking ability. Therefore, the calibrated and uncalibrated ROC-AUC values should remain nearly identical.

During development, an earlier pipeline using **XGBoost + `CalibratedClassifierCV(cv=5)`** produced calibrated ROC-AUC values between **0.54–0.66**, indicating that internal cross-validation refitting significantly degraded discrimination on the small, imbalanced dataset.

After extensive debugging, the final solution adopted a **Random Forest** model with isotonic calibration. This preserved discrimination while substantially improving probability calibration. The complete development process is documented in `notebooks/04_modeling_final.ipynb`.

---

## Operating Threshold Selection

Because the dataset is highly imbalanced (approximately **90.4% successful launches**), using the default classification threshold of **0.5** is inappropriate. Instead, the operating threshold was selected by maximizing the **F1-score for the failure class**.

| Failure Probability Threshold | Precision | Recall |
| :---: | :---: | :---: |
| **0.241 (Selected)** | **0.48** | **0.33** |

### Classification Report

```text
              precision    recall    f1-score    support

Failure          0.48       0.33       0.40         48
Success          0.96       0.98       0.97        792

Accuracy                               0.94        840
Macro Avg        0.72       0.66       0.68        840
Weighted Avg     0.93       0.94       0.94        840
```

At this threshold:

- When the model predicts a launch as **high risk**, it is correct **48%** of the time.
- This is approximately **8.4× higher** than the baseline failure rate (5.7%).
- The model identifies roughly **one-third of historical failures**, providing a practical screening tool despite having access only to historical mission metadata.

The selected threshold is stored in:

```text
models/model_metadata.json
```

under the key:

```text
chosen_threshold_failure_prob
```

and is used by `src/predict.py`.

---

## Baseline Comparison

| Model | ROC-AUC | Brier Score | Notes |
| :--- | :---: | :---: | :--- |
| Logistic Regression | 0.72–0.73 | 0.157 | `class_weight="balanced"` |
| XGBoost | 0.7364 | 0.1581 | Regularization tuned |
| XGBoost + Isotonic Calibration | 0.5364 | 0.0547 | Rejected due to discrimination collapse |
| **Random Forest + Isotonic Calibration** | **0.7489** | **0.0472** | **Final selected model** |

---

## Calibration Curve

See:

```text
presentation/calibration_curve.png
```

The calibrated probabilities closely follow the ideal diagonal, with a **Mean Calibration Error (MCE) of 0.037**, indicating that predicted probabilities differ from observed frequencies by only about **3.7 percentage points** on average.

---

## Precision–Recall Curve

See:

```text
presentation/pr_curve_failure.png
```

---

## SHAP Global Feature Importance

See:

```text
presentation/shap_global_importance.png
```

The most influential predictors include:

- `vehicle_prior_success_rate`
- `site_prior_success_rate`

These features align with the established aerospace observation that launch reliability generally improves as launch vehicles and launch sites mature.

---

## Cross-Validation

A **5-fold `TimeSeriesSplit`** evaluation was performed to ensure the reported results were not dependent on a single chronological split.

Detailed fold-wise results are available in:

```text
notebooks/04_modeling_final.ipynb
```

---

## Reproducing the Results

Clone the repository:

```bash
git clone https://github.com/DhairyaDave08/BetaData-SpaceX.git
cd BetaData-SpaceX
pip install -r requirements.txt
```

Run the notebook:

```text
notebooks/04_modeling_final.ipynb
```

Alternatively, the trained model is already available at:

```text
models/model.pkl
```

and can be loaded directly using:

```text
src/predict.py
```
