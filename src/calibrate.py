

import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss


def calibrate_model(pipeline, X_train, y_train, method: str = "isotonic", cv: int = 5):
    """
    Wraps a fitted or unfitted pipeline with calibration.
    Isotonic is preferred over Platt/sigmoid here because it's non-parametric
    and handles irregularities in a small, non-uniform dataset better than
    assuming a sigmoid-shaped miscalibration.
    """
    calibrated = CalibratedClassifierCV(pipeline, method=method, cv=cv)
    calibrated.fit(X_train, y_train)
    return calibrated


def get_calibration_curve(y_true, probs, n_bins: int = 10):
    fraction_pos, mean_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({
        "mean_predicted_prob": mean_pred,
        "fraction_of_positives": fraction_pos,
    })


def report_calibration_quality(y_true, probs) -> dict:
    brier = brier_score_loss(y_true, probs)
    curve = get_calibration_curve(y_true, probs)
    # Simple calibration error: average absolute gap between predicted and actual
    calibration_error = np.mean(np.abs(curve["mean_predicted_prob"] - curve["fraction_of_positives"]))
    return {
        "brier_score": brier,
        "mean_calibration_error": calibration_error,
        "curve": curve,
    }
