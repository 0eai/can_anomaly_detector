# config.py
# Defines configuration structure and paths

import os

DATA_DIR = '../datasets/10) CAN-Intrusion Dataset' # IMPORTANT: Replace this path
if not os.path.isdir(DATA_DIR):
    print(f"WARNING: Data directory not found at '{DATA_DIR}'. Please set the correct path.")
    raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

NORMAL_DATA_FILENAME = 'Attack_free_dataset.txt'
ATTACK_FILENAMES = [
    'DoS_attack_dataset.txt',
    'Fuzzy_attack_dataset.txt',
    'Impersonation_attack_dataset.txt'
]

MODEL_SAVE_DIR = "saved_models" # Or specify a subdirectory like "saved_models"
SCALER_PATH = os.path.join(MODEL_SAVE_DIR, 'scaler.joblib')
RNN_AE_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'rnn_ae_model.keras')
RNN_AE_THRESHOLD_PATH = os.path.join(MODEL_SAVE_DIR, 'rnn_ae_threshold.joblib')
LODA_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'loda_model.joblib') # Placeholder path
LODA_THRESHOLD_PATH = os.path.join(MODEL_SAVE_DIR, 'loda_threshold.joblib') # Placeholder path
OCSVM_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'ocsvm_model.joblib')
OCSVM_THRESHOLD_PATH = os.path.join(MODEL_SAVE_DIR, 'ocsvm_threshold.joblib')

# --- Core Configuration Dictionary Structure ---
DEFAULT_CONFIG = {
    'data_paths': {
        'data_dir': DATA_DIR,
        'normal_filename': NORMAL_DATA_FILENAME,
        'attack_filenames': ATTACK_FILENAMES,
    },
    'model_paths': {
        'scaler': SCALER_PATH,
        'rnn_ae': RNN_AE_MODEL_PATH, 'rnn_ae_threshold': RNN_AE_THRESHOLD_PATH,
        'loda': LODA_MODEL_PATH, 'loda_threshold': LODA_THRESHOLD_PATH,
        'ocsvm': OCSVM_MODEL_PATH, 'ocsvm_threshold': OCSVM_THRESHOLD_PATH,
    },
    'static_checks': {
        'known_ids': set(), # Populated by profiler
        'expected_dlc': {}  # Populated by profiler
    },
    'feature_config': {
        'normal_inter_arrival': {}, # Populated by profiler
        'normal_nonzero_bytes': {}, # Populated by profiler
        'impute_timing_value': 0.01 # Default, overwritten by profiler
    },
    'ml_models': {
        'rnn_ae': {
            'params': { 'encoding_dim': 1, 'epochs': 2, 'batch_size': 64,
                        'validation_split': 0.1 },
            'threshold_percentile': 95.0 # Outside params
        },
        'loda': {
            'params': { # Only LODA init params here
                'bins': 10,
                'n_estimators': 100,
                # 'random_state': 42, # Set via training_options if needed
            },
            'threshold_percentile': 95.0 # <<< OUTSIDE 'params' >>>
        },
        'ocsvm': { 'params': { 'kernel': 'rbf', # or 'linear' for speed
                                'gamma': 'scale',
                                'nu': 0.05,       # Expected anomaly fraction (tune this)
                                'cache_size': 100000 # Increase cache
                            },
                    'threshold_percentile': 95.0 # Percentile for thresholding OCSVM scores
                },
    },
    'anomaly_analyzer': {
        'min_models_triggered': 2 # Default trigger level
    },
    'learning': { # Parameters controlling the profiling stage
        'timing_percentiles': (5.0, 95.0),
        'payload_percentiles': (5.0, 95.0)
    },
    'training_options': { # Options for training phase
        'ocsvm_subsample_max': 100000, # Max samples for OCSVM if used
        'random_seed': 42 # For reproducibility in sampling/training
    }
}

def get_config():
    """Returns the default configuration dictionary."""
    if MODEL_SAVE_DIR != "." and not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        print(f"Created model save directory: {MODEL_SAVE_DIR}")
    return DEFAULT_CONFIG