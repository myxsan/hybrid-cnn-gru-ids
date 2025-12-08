# model/ids_model.py
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn


class CNN_GRU_IDS(nn.Module):
    def __init__(self):
        super(CNN_GRU_IDS, self).__init__()

        # Spatial Feature Extraction (CNN)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.relu = nn.ReLU()

        # Pooling & Normalization
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.batch_norm = nn.BatchNorm1d(64)

        # Temporal Feature Extraction (GRU)
        self.gru = nn.GRU(input_size=64, hidden_size=128, batch_first=True)

        # Classification Head
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 2) # Binary: 0=Benign, 1=Attack

    def forward(self, x):
        # 1. CNN Phase
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.batch_norm(x)

        # 2. Reshape for GRU
        x = x.permute(0, 2, 1)

        # 3. GRU Phase
        _, h_n = self.gru(x)
        x = h_n.squeeze(0)

        # 4. Dense Phase
        x = self.dropout(x)
        x = self.fc(x)
        return x



BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"


class IDSService:
    """
    Wraps:
    - Model loading
    - Scaler loading
    - Feature ordering
    - Single prediction
    """
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

        # 1) Load feature order
        feature_file = ARTIFACTS_DIR / "feature_order.json"
        with feature_file.open() as f:
            self.feature_order = json.load(f)

        # 2) Load scaler
        self.scaler = joblib.load(ARTIFACTS_DIR / "cic_scaler.pkl")

        # 3) Load model
        self.model = CNN_GRU_IDS().to(self.device)
        state_dict = torch.load(
            ARTIFACTS_DIR / "cnn_gru_ids_cic.pth",
            map_location=self.device,
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _dict_to_array(self, feature_dict: dict) -> np.ndarray:
        """
        Turn {feature_name: value} → numpy array in correct order & shape.
        """
        vals = [feature_dict[name] for name in self.feature_order]
        arr = np.array(vals, dtype=np.float32).reshape(1, -1)  # (1, n_features)
        # scale with CIC scaler
        arr_scaled = self.scaler.transform(arr)
        return arr_scaled  # still (1, n_features)

    def predict_single(self, feature_dict: dict) -> dict:
        """
        Do a single prediction on one flow.
        Returns:
          {
            "pred_class": int,
            "probabilities": [p0, p1, ...]
          }
        """
        arr_scaled = self._dict_to_array(feature_dict)

        # >>> VERY IMPORTANT: shape this exactly like in training <<<
        # In many tabular-CNN/GRU setups it’s (batch, seq_len, features).
        # Example if you used (batch, 1, n_features) in training:
        x = torch.from_numpy(arr_scaled).unsqueeze(1).to(self.device)  # (1, 1, n_features)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class = int(np.argmax(probs))

        return {
            "pred_class": pred_class,
            "probabilities": probs.tolist(),
        }
