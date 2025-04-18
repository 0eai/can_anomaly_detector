# ml_models.py
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import os
import traceback
# Import sklearn model for type checking if needed
from sklearn.svm import OneClassSVM as SklearnOCSVMClass

try:
    from anlearn.loda import LODA as AnlearnLODAClass
    ANLEARN_LODA_LOADED = True
except ImportError:
    AnlearnLODAClass = None
    ANLEARN_LODA_LOADED = False
    # print("Warning: anlearn.loda not imported.") # Reduce verbosity

class BaseMLModel:
    """Base class for inference models."""
    def __init__(self, model_name, config):
        # Set attributes first
        self.model_name = model_name # Still set the attribute on self
        # Store paths directly on self - these might be needed by subclasses later?
        # It's generally better practice for _load to get them from config if needed.
        # Let's keep using config within _load for path resolution.
        self.model = None
        self.scaler = None
        self.threshold = None

        # <<< Pass model_name and config explicitly to _load >>>
        # This ensures _load receives them as arguments, independent of self state
        self._load(model_name, config)

    # <<< Update _load signature to accept arguments >>>
    def _load(self, model_name, config):
        """Loads the trained model, scaler, and OPTIONAL threshold."""
        # Use the passed model_name argument directly for loading logic/error messages
        current_model_name = model_name # Use argument, not self.model_name initially
        paths = config.get('model_paths', {}) # Get paths from config argument

        model_loaded = False
        scaler_loaded = False
        threshold_loaded = True # Assume true if threshold not needed (e.g., LODA)

        try:
            # --- Load Scaler ---
            scaler_path = paths.get('scaler') # Use path from arg's config
            if scaler_path and os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path); scaler_loaded = True
                # print(f"DEBUG: Scaler loaded for {current_model_name}")
            else:
                print(f"ERROR: Scaler path missing/invalid: {scaler_path}. Model '{current_model_name}' disabled.")
                self.model = None; return # Stop loading if scaler fails

            # --- Load Model ---
            model_path = paths.get(current_model_name) # Use path from arg's config
            if model_path and os.path.exists(model_path):
                # Specific loading logic using current_model_name
                if current_model_name == 'rnn_ae':
                    self.model = keras.models.load_model(model_path)
                    print(f"INFO: Loaded Keras model for {current_model_name}")
                    model_loaded = True
                elif current_model_name == 'loda':
                    self.model = joblib.load(model_path)
                    if isinstance(self.model, dict) and "Placeholder" in self.model.get("info",""): print(f"INFO: Loaded placeholder LODA: {self.model}")
                    elif AnlearnLODAClass and isinstance(self.model, AnlearnLODAClass): print(f"INFO: Loaded anlearn LODA model")
                    else: print(f"Warning: LODA object type unrecognized."); self.model = None
                    model_loaded = self.model is not None
                elif current_model_name == 'ocsvm':
                    self.model = joblib.load(model_path); print(f"INFO: Loaded OCSVM model"); model_loaded = True
                else:
                    self.model = joblib.load(model_path); print(f"INFO: Loaded generic model for {current_model_name}"); model_loaded = True
            else: print(f"ERROR: Model path missing/invalid: {model_path}. Model '{current_model_name}' disabled."); self.model = None; return

            # --- Load Threshold (If applicable) ---
            if current_model_name != 'loda': # LODA doesn't need separate threshold file
                threshold_path = paths.get(f"{current_model_name}_threshold") # Use path from arg's config
                if threshold_path and os.path.exists(threshold_path):
                    self.threshold = joblib.load(threshold_path)
                    threshold_loaded = True
                    print(f"INFO: Loaded threshold for {current_model_name}: {self.threshold:.6f}")
                else:
                    print(f"ERROR: Threshold path missing/invalid: {threshold_path}. Model '{current_model_name}' disabled.")
                    threshold_loaded = False # Mark as failed

            # Final check
            if not (model_loaded and scaler_loaded and threshold_loaded):
                print(f"ERROR: Failed loading some components for {current_model_name}. Model disabled.")
                self.model = None # Ensure model is disabled

        except Exception as e:
            # Use current_model_name from argument for error message
            print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"ERROR: Exception during loading process for model '{current_model_name}'.") # Use arg
            print(f"  Exception Type: {type(e).__name__}")
            print(f"  Exception Args: {e.args}")
            # import traceback # Moved import to top
            print("  Traceback:")
            traceback.print_exc()
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            # Ensure model attribute on self is None if loading fails
            self.model = None


    def get_anomaly_score(self, features_array): raise NotImplementedError
    def predict_anomaly(self, features_array):
        if self.model is None or self.scaler is None or self.threshold is None: return -1
        score = self.get_anomaly_score(features_array)
        if score == -1: return -1
        return 1 if score > self.threshold else 0

# --- Specific Model Implementations ---

class RNN_AE_Model(BaseMLModel):
    """Autoencoder (Replicator NN) Inference Model."""
    def get_anomaly_score(self, features_array):
        """Calculates reconstruction error (MSE) as anomaly score."""
        if self.model is None or self.scaler is None: return -1
        try:
            features_scaled = self.scaler.transform(features_array.astype(np.float32).reshape(1, -1))
            reconstructed = self.model.predict(features_scaled, verbose=0)
            mse = np.mean(np.power(features_scaled - reconstructed, 2), axis=1)
            return float(mse[0])
        except Exception: return -1

class LODA_Model(BaseMLModel):
    """LODA Inference Model using anlearn."""
    def get_anomaly_score(self, features_array):
        """Calculates LODA score (inverted score_samples, higher=anomaly)."""
        is_placeholder = isinstance(self.model, dict) and "Placeholder" in self.model.get("info","")
        if self.model is None or self.scaler is None or is_placeholder:
            if is_placeholder: return random.uniform(0.0, 1.5) * float(self.threshold if self.threshold is not None else 1.0)
            else: return -1 # Disabled or not loaded

        try:
            features_array_float = features_array.astype(np.float32)
            features_scaled = self.scaler.transform(features_array_float.reshape(1, -1))

            # <<< --- Use score_samples() and invert --- >>>
            # Assume score_samples exists and lower = more normal
            if hasattr(self.model, 'score_samples'):
                raw_score = self.model.score_samples(features_scaled)
                # Invert the score so higher is more anomalous
                anomaly_score = -raw_score
                return float(anomaly_score[0]) # Return the single inverted score
            else:
                print(f"ERROR: Loaded LODA model object does not have 'score_samples' method.")
                return -1 # Error state if method missing

        except AttributeError: # Catch if method doesn't exist on loaded object
            print(f"ERROR: LODA model missing 'score_samples' method during inference.")
            return -1
        except Exception as e:
            print(f"Error LODA scoring: {e}")
            return -1

class OCSVM_Model(BaseMLModel):
    """OneClassSVM Inference Model."""
    def get_anomaly_score(self, features_array):
        """Calculates OCSVM score (inverted decision function, higher=anomaly)."""
        if self.model is None or self.scaler is None: return -1
        try:
            features_scaled = self.scaler.transform(features_array.astype(np.float32).reshape(1, -1))
            # decision_function: higher score is more normal (further inside boundary)
            # Invert it so higher score is more anomalous
            score = -self.model.decision_function(features_scaled)
            return float(score[0])
        except Exception as e:
            # print(f"Error during OCSVM scoring: {e}") # Reduce verbosity
            return -1