# components.py
# Defines StaticChecker and AnomalyAnalyzer classes.

class StaticChecker:
    """Performs static checks based on learned profile."""
    def __init__(self, static_config):
        self.config = static_config if isinstance(static_config, dict) else {}
        self.known_ids = self.config.get('known_ids', set())
        self.expected_dlcs = self.config.get('expected_dlc', {})

    def check(self, message):
        """Checks a single CAN message. Returns (is_anomaly, description)."""
        if message.can_id not in self.known_ids:
            return True, f"Static Fail: Unknown ID 0x{message.can_id:X}"

        expected_dlc = self.expected_dlcs.get(message.can_id)
        if expected_dlc is not None and message.dlc != expected_dlc:
            return True, f"Static Fail: Bad DLC for 0x{message.can_id:X} (Got {message.dlc}, Exp {expected_dlc})"

        # Add more static checks here if needed (e.g., basic timing bounds from profile)

        return False, "Static OK"


class AnomalyAnalyzer:
    """Analyzes ML model anomaly scores against thresholds."""
    def __init__(self, ml_model_names, analyzer_config):
        """
        Args:
            ml_model_names (list): List of names of the active ML models (e.g., ['rnn_ae', 'loda']).
            analyzer_config (dict): Configuration for the analyzer, expects 'min_models_triggered'.
        """
        self.min_models_triggered = analyzer_config.get('min_models_triggered', 1)
        self.model_names = ml_model_names if isinstance(ml_model_names, list) else []

    def analyze(self, model_outputs):
        """
        Analyzes scores {model_name: {'score': float, 'threshold': float}}.
        Returns (is_anomaly, description_string).
        """
        triggered_count = 0
        details = []
        error_models = 0
        warning_models = 0 # Count models that didn't produce valid output

        if not model_outputs:
            return False, "ML OK: No model outputs."

        for model_name in self.model_names: # Iterate through expected models
            output_data = model_outputs.get(model_name)

            if output_data is None:
                # print(f"Warning: No output found for model {model_name}")
                warning_models += 1
                continue

            score = output_data.get('score')
            threshold = output_data.get('threshold')

            if score is None or threshold is None:
                # print(f"Warning: Missing score or threshold for model {model_name}")
                warning_models += 1
                continue

            if score == -1: # Specific error code from model's get_anomaly_score
                error_models += 1
                # Optionally add detail: details.append(f"{model_name}:Error")
                continue

            # Compare score against threshold
            if score > threshold:
                triggered_count += 1
                # Include score and threshold in details for context
                details.append(f"{model_name}:{score:.3f}>{threshold:.3f}")

        desc_suffix = ""
        if error_models > 0: desc_suffix += f" Errors:{error_models}"
        if warning_models > 0: desc_suffix += f" Warnings:{warning_models}"


        if triggered_count >= self.min_models_triggered:
            return True, f"ML Fail:{triggered_count}[{','.join(details)}]" + desc_suffix
        else:
            # Optionally show non-triggering scores for debugging
            ok_details = [f"{name}:{d['score']:.3f}" for name, d in model_outputs.items() if name in self.model_names and d.get('score', -1) != -1 and d.get('threshold') is not None and d['score'] <= d['threshold']]
            # return False, f"ML OK:{triggered_count} ({','.join(ok_details)})" + desc_suffix
            return False, f"ML OK:{triggered_count}" + desc_suffix