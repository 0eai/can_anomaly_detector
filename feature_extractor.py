# feature_extractor.py
# Defines the FeatureExtractor class.

import numpy as np

class FeatureExtractor:
    """Extracts features from CAN messages for ML models."""
    def __init__(self, feature_config):
        self.config = feature_config if isinstance(feature_config, dict) else {}
        self.last_timestamps = {}
        # Get the imputation value learned during profiling or use default
        self.impute_value = self.config.get('impute_timing_value', 0.01)
        if self.impute_value <= 0: # Ensure imputation value is positive
            # print("Warning: Learned impute value <= 0, using default 0.01")
            self.impute_value = 0.01

    def extract(self, message):
        """
        Extracts features from a CANMessage object.
        Returns a 1D NumPy array: [inter_arrival_time, non_zero_bytes].
        """
        can_id = message.can_id
        current_timestamp = message.timestamp

        # 1. Inter-arrival time
        prev_ts = self.last_timestamps.get(can_id)
        if prev_ts is not None:
            inter_arrival = current_timestamp - prev_ts
            # Handle potential clock issues or duplicate timestamps
            if inter_arrival < 0:
                inter_arrival = 0 # Treat negative delta as zero delta
        else:
            # Impute the value for the first message seen for this ID
            inter_arrival = self.impute_value

        # Update state *after* calculation for next message
        self.last_timestamps[can_id] = current_timestamp

        # 2. Basic Payload Feature: Count non-zero bytes
        non_zero = sum(1 for byte in message.data if byte != 0)

        # Return features as a fixed-order NumPy array
        return np.array([inter_arrival, non_zero], dtype=float)

    def reset_state(self):
        """Resets the internal state (last timestamps)."""
        self.last_timestamps = {}