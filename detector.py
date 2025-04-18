# detector.py
# Defines the HybridAnomalyDetector class.

# Import necessary components from other modules
from components import StaticChecker, AnomalyAnalyzer
from feature_extractor import FeatureExtractor
from ml_models import RNN_AE_Model, LODA_Model # Import actual/placeholder models
# from ml_models import OCSVM_Model # Import if using OCSVM

class HybridAnomalyDetector:
    """Orchestrates the hybrid detection process using trained models."""
    def __init__(self, config):
        self.config = config
        # Initialize components using relevant config sections
        self.static_checker = StaticChecker(config.get('static_checks', {}))
        self.feature_extractor = FeatureExtractor(config.get('feature_config', {}))

        self.ml_models = {}
        ml_config = config.get('ml_models', {})
        # Instantiate ML model inference classes
        # The classes themselves handle loading models from paths specified in config
        if 'rnn_ae' in ml_config:
            self.ml_models['rnn_ae'] = RNN_AE_Model('rnn_ae', config)
        if 'loda' in ml_config:
            self.ml_models['loda'] = LODA_Model('loda', config)
        # if 'ocsvm' in ml_config:
        #      self.ml_models['ocsvm'] = OCSVM_Model('ocsvm', config)

        active_model_names = list(self.ml_models.keys())
        print(f"Initialized {len(active_model_names)} ML models for inference: {active_model_names}")

        self.anomaly_analyzer = AnomalyAnalyzer(active_model_names, config.get('anomaly_analyzer', {}))
        self.results = [] # Store results for evaluation

    def log_detection(self, message, is_anomaly, reason):
        """Stores the detection result."""
        self.results.append({
            'timestamp': message.timestamp,
            'can_id': message.can_id,
            'true_label': message.label,
            'predicted_label': 'Anomaly' if is_anomaly else 'Normal',
            'reason': reason
        })

    def process_message(self, message):
        """Processes a single CAN message through the pipeline."""
        # 1. Static Check
        is_static_anomaly, static_desc = self.static_checker.check(message)
        if is_static_anomaly:
            self.log_detection(message, True, static_desc)
            return # Stop processing if static check fails

        # 2. Feature Extraction
        # Extract returns a NumPy array [time, nonzero]
        features_array = self.feature_extractor.extract(message)
        if features_array is None: # Should not happen with current logic, but check
            print(f"Warning: Feature extraction failed for msg at {message.timestamp}")
            self.log_detection(message, False, "FeatureExtractionError") # Log as Normal or specific error?
            return

        # 3. ML Model Inference (Get Scores)
        model_outputs = {}
        if self.ml_models:
            for name, model in self.ml_models.items():
                # The model instance handles loading errors internally
                # We need score AND threshold for the analyzer
                score = model.get_anomaly_score(features_array)
                threshold = model.threshold # Threshold loaded during model init
                model_outputs[name] = {'score': score, 'threshold': threshold}
        else:
            # If no ML models, treat as Normal for ML stage
            is_ml_anomaly, ml_desc = False, "Normal (No ML)"
            self.log_detection(message, is_ml_anomaly, ml_desc)
            return


        # 4. Anomaly Analysis (Compare scores to thresholds)
        is_ml_anomaly, ml_desc = self.anomaly_analyzer.analyze(model_outputs)

        # 5. Log Final Result
        if is_ml_anomaly:
            self.log_detection(message, True, ml_desc)
        else:
            # Passed static and ML checks
            self.log_detection(message, False, "Normal")

    def get_results(self):
        """Returns the accumulated detection results."""
        return self.results

    def reset_state(self):
        """Resets internal state (feature extractor and results)."""
        self.feature_extractor.reset_state()
        self.results = []