import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib, os

# 20 meaningful features with defaults and descriptions
FEATURE_SCHEMA = [
    ("amount",            0.0,  "Số tiền giao dịch (VND)"),
    ("time_hour",        12.0,  "Giờ giao dịch (0-23)"),
    ("day_of_week",       2.0,  "Thứ trong tuần (0=T2, 6=CN)"),
    ("merchant_category", 0.0,  "Loại merchant (0-9)"),
    ("location_risk",     0.0,  "Rủi ro vị trí (0-1)"),
    ("device_risk",       0.0,  "Rủi ro thiết bị (0-1)"),
    ("channel_risk",      0.0,  "Rủi ro kênh giao dịch (0-1)"),
    ("is_new_device",     0.0,  "Thiết bị mới (0/1)"),
    ("is_new_location",   0.0,  "Vị trí mới (0/1)"),
    ("transaction_freq",  1.0,  "Tần suất GD gần đây (số GD/ngày)"),
    ("avg_amount_7d",     0.0,  "Số tiền TB 7 ngày qua (VND)"),
    ("amount_deviation",  0.0,  "Độ lệch so với TB (VND)"),
    ("recent_fraud_count",0.0,  "Số cảnh báo fraud gần đây"),
    ("distance_km",       0.0,  "Khoảng cách từ vị trí thường (km)"),
    ("is_weekend",        0.0,  "Cuối tuần (0/1)"),
    ("is_night",          0.0,  "Ban đêm 22-5h (0/1)"),
    ("velocity_1h",       0.0,  "Số GD trong 1h qua"),
    ("velocity_24h",      0.0,  "Số GD trong 24h qua"),
    ("cross_border",      0.0,  "Giao dịch quốc tế (0/1)"),
    ("card_present",      1.0,  "Thẻ vật lý có mặt (0/1)"),
]

FEATURE_NAMES = [f[0] for f in FEATURE_SCHEMA]
FEATURE_DEFAULTS = [f[1] for f in FEATURE_SCHEMA]
FEATURE_DESCRIPTIONS = {f[0]: f[2] for f in FEATURE_SCHEMA}
NUM_RAW_FEATURES = len(FEATURE_SCHEMA)  # 20
NUM_MODEL_FEATURES = 30


def extract_features(input_dict):
    """Extract meaningful features from input dict, filling defaults for missing."""
    return [float(input_dict.get(name, default)) for name, default, _ in FEATURE_SCHEMA]


def engineer_features(raw_features):
    """Transform 20 raw features → 30 model features via interaction terms.

    20 base features + 10 engineered interactions:
      0-19: raw features
      20: amount × location_risk
      21: amount × channel_risk
      22: is_weekend × is_night
      23: velocity_1h × amount
      24: amount_deviation × location_risk
      25: is_new_device × channel_risk
      26: transaction_freq × velocity_24h
      27: amount / (avg_amount_7d + 1)
      28: amount × is_new_location
      29: recent_fraud_count × amount_deviation
    """
    f = np.array(raw_features, dtype=float)
    interactions = np.array([
        f[0] * f[4],              # amount × location_risk
        f[0] * f[6],              # amount × channel_risk
        f[14] * f[15],            # is_weekend × is_night
        f[16] * f[0],             # velocity_1h × amount
        f[11] * f[4],             # amount_deviation × location_risk
        f[7] * f[6],              # is_new_device × channel_risk
        f[9] * f[17],             # transaction_freq × velocity_24h
        f[0] / (abs(f[10]) + 1),  # amount / (avg_amount_7d + 1)
        f[0] * f[8],              # amount × is_new_location
        f[12] * f[11],            # recent_fraud_count × amount_deviation
    ])
    return np.concatenate([f, interactions])


class FeaturePipeline:
    """Manages feature extraction, engineering, and scaling."""

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.scaler_path = os.path.join(model_dir, "scaler.pkl")
        self.scaler = None
        self._load_scaler()

    def _load_scaler(self):
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
            print(f"[+] Scaler loaded from {self.scaler_path}")

    def transform(self, raw_features):
        """Full pipeline: extract → engineer → scale → return 30-dim vector."""
        engineered = engineer_features(raw_features).reshape(1, -1)
        if self.scaler:
            engineered = self.scaler.transform(engineered)
        return engineered.flatten()

    def fit_scaler(self, feature_matrix):
        """Fit scaler on an array of engineered feature vectors (n_samples × 30)."""
        self.scaler = StandardScaler()
        self.scaler.fit(feature_matrix)
        joblib.dump(self.scaler, self.scaler_path)
        print(f"[+] Scaler fitted and saved to {self.scaler_path}")

    @property
    def is_fitted(self):
        return self.scaler is not None
