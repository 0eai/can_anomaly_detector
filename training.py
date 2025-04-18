# training.py
import time
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# <<< Import OCSVM >>>
from sklearn.svm import OneClassSVM

# Import LODA placeholder/actual
try:
    from anlearn.loda import LODA
    ANLEARN_LODA_LOADED = True
except ImportError:
    LODA = None # Define as None if import fails
    ANLEARN_LODA_LOADED = False

# Local imports
from data_parser import load_and_parse_data
from feature_extractor import FeatureExtractor

def train_ml_models(normal_data_file, config):
    """Loads normal data, extracts features, trains AE, LODA, and OCSVM."""
    print("\n--- Training ML Models ---")
    model_paths = config['model_paths']
    ml_config = config['ml_models']
    feature_config = config['feature_config']
    training_options = config.get('training_options', {})
    random_seed = training_options.get('random_seed', None)


    # 1. Load normal data and extract features (Same as before)
    print("Loading normal data for training...")
    training_messages = load_and_parse_data(normal_data_file, 'Normal')
    if not training_messages: print("ERROR: No training data."); return False
    print("Extracting features...")
    feature_extractor = FeatureExtractor(feature_config)
    features_list = []; feature_extractor.reset_state()
    for msg in training_messages: features_list.append(feature_extractor.extract(msg))
    if not features_list: print("ERROR: No features extracted."); return False
    X_train_all = np.array(features_list, dtype=np.float32)
    print(f"Extracted {X_train_all.shape[0]} feature vectors.")

    # 2. Fit Scaler (Same as before)
    print("Fitting StandardScaler...")
    scaler = StandardScaler().fit(X_train_all)
    joblib.dump(scaler, model_paths['scaler'])
    print(f"Scaler saved to {model_paths['scaler']}")
    X_train_all_scaled = scaler.transform(X_train_all)

    # Split data once for validation/threshold setting
    # Use a split fraction appropriate for all models needing it
    val_split_fraction = 0.1 # Example, can be configured
    if X_train_all_scaled.shape[0] > 1 / val_split_fraction:
        X_train_main, X_val = train_test_split(
            X_train_all_scaled, test_size=val_split_fraction, random_state=random_seed
        )
        print(f"Split data: Training {X_train_main.shape[0]}, Validation/Threshold {X_val.shape[0]}")
    else:
        X_train_main = X_train_all_scaled
        X_val = X_train_all_scaled # Use all data if too small to split
        print(f"Using all {X_train_main.shape[0]} samples for training and thresholding (dataset too small for split).")


    # --- Autoencoder (RNN_AE) Training ---
    if 'rnn_ae' in ml_config:
        # ... (AE training logic remains the same, using X_train_main, X_val) ...
        print(f"\n--- Training Autoencoder (RNN_AE) model ---")
        ae_params = ml_config['rnn_ae']['params']
        n_features = X_train_all_scaled.shape[1]
        encoding_dim = ae_params.get('encoding_dim', max(1, n_features // 2))
        input_layer = keras.Input(shape=(n_features,))
        encoded = layers.Dense(max(1, n_features * 2), activation='relu')(input_layer)
        encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
        decoded = layers.Dense(max(1, n_features * 2), activation='relu')(encoded)
        decoded = layers.Dense(n_features, activation='linear')(decoded)
        autoencoder = keras.Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        epochs=ae_params.get('epochs', 20); batch_size=ae_params.get('batch_size', 64)
        start_ae_time = time.time()
        print(f"Starting AE training: Epochs={epochs}, BatchSize={batch_size}")
        history = autoencoder.fit(X_train_main, X_train_main, epochs=epochs, batch_size=batch_size,
                                shuffle=True, validation_data=(X_val, X_val), verbose=2)
        end_ae_time = time.time()
        print(f"AE training finished in {end_ae_time - start_ae_time:.2f} seconds.")
        autoencoder.save(model_paths['rnn_ae']); print(f"AE model saved to {model_paths['rnn_ae']}")
        print("Calculating AE reconstruction errors...")
        reconstructed_X = autoencoder.predict(X_val)
        mse = np.mean(np.power(X_val - reconstructed_X, 2), axis=1)
        threshold_percentile = ae_params.get('threshold_percentile', 95.0)
        threshold = np.percentile(mse, threshold_percentile)
        print(f"AE Anomaly threshold ({threshold_percentile}th percentile): {threshold:.6f}")
        joblib.dump(threshold, model_paths['rnn_ae_threshold'])
        print(f"AE threshold saved to {model_paths['rnn_ae_threshold']}")


    # --- LODA Training ---
    if 'loda' in ml_config:
        if ANLEARN_LODA_LOADED:
            # ... (parameter setup, LODA instantiation, fitting, saving model) ...
            print(f"\n--- Training LODA model ---")
            loda_config = ml_config['loda']
            loda_params = loda_config.get('params', {}).copy()
            threshold_percentile = loda_config.get('threshold_percentile', 95.0)
            if random_seed is not None: loda_params['random_state'] = random_seed
            print(f"Initializing anlearn LODA with params: {loda_params}")
            loda_model = LODA(**loda_params)
            print(f"Fitting LODA on {X_train_main.shape[0]} samples...")
            start_loda_time = time.time(); loda_model.fit(X_train_main); end_loda_time = time.time()
            print(f"LODA fitting finished in {end_loda_time - start_loda_time:.2f} seconds.")
            joblib.dump(loda_model, model_paths['loda']); print(f"LODA model saved to {model_paths['loda']}")


            # <<< --- Determine anomaly score threshold using score_samples --- >>>
            print("Calculating LODA scores on validation data using score_samples()...")
            # Assume score_samples() returns scores where lower = more normal
            # E.g., negative log-likelihood
            try:
                val_raw_scores = loda_model.score_samples(X_val)
                # Invert scores so higher = more anomalous
                anomaly_scores = -val_raw_scores

                threshold = np.percentile(anomaly_scores, threshold_percentile)
                print(f"LODA Anomaly threshold ({threshold_percentile}th percentile of inverted scores): {threshold:.6f}")

                # Save the threshold
                joblib.dump(threshold, model_paths['loda_threshold'])
                print(f"LODA threshold saved to {model_paths['loda_threshold']}")

            except AttributeError:
                print("ERROR: 'LODA' object has no attribute 'score_samples' either. Cannot determine threshold.")
                print("INFO: Saving placeholder LODA threshold.")
                joblib.dump(1.0, model_paths['loda_threshold']) # Save placeholder
                # Optionally return False here if thresholding is critical
            except Exception as e:
                print(f"ERROR during LODA threshold calculation: {e}")
                import traceback
                traceback.print_exc()
                print("INFO: Saving placeholder LODA threshold.")
                joblib.dump(1.0, model_paths['loda_threshold']) # Save placeholder


        else: # Placeholder logic if anlearn not loaded
            print("\n--- LODA Training (Placeholder - anlearn not installed) ---")
            joblib.dump({"info": "Placeholder - anlearn not installed"}, model_paths['loda'])
            joblib.dump(1.0, model_paths['loda_threshold'])
            print(f"Placeholder LODA model and threshold saved.")


    # --- *** OCSVM Training *** ---
    if 'ocsvm' in ml_config:
        print("\n--- Training OCSVM model ---")
        ocsvm_params = ml_config['ocsvm']['params'].copy()
        threshold_percentile = ml_config['ocsvm'].get('threshold_percentile', 95.0) # Get threshold percentile

        # Apply Subsampling for OCSVM Training Data
        max_ocsvm_samples = training_options.get('ocsvm_subsample_max', None)
        X_ocsvm_train_input = X_train_main # Start with main training data

        if max_ocsvm_samples and X_train_main.shape[0] > max_ocsvm_samples:
            print(f"Subsampling OCSVM training data from {X_train_main.shape[0]} to {max_ocsvm_samples} samples...")
            if random_seed is not None: np.random.seed(random_seed) # Seed sampling
            indices = np.random.choice(X_train_main.shape[0], max_ocsvm_samples, replace=False)
            X_ocsvm_train_input = X_train_main[indices]

        print(f"Training OneClassSVM on {X_ocsvm_train_input.shape[0]} samples (params: {ocsvm_params})...")
        ocsvm_model = OneClassSVM(**ocsvm_params)
        start_ocsvm_time = time.time()
        ocsvm_model.fit(X_ocsvm_train_input) # Fit on potentially subsampled data
        end_ocsvm_time = time.time()
        print(f"OCSVM training finished in {end_ocsvm_time - start_ocsvm_time:.2f} seconds.")

        # Save the OCSVM model
        joblib.dump(ocsvm_model, model_paths['ocsvm'])
        print(f"OCSVM model saved to {model_paths['ocsvm']}")

        # Determine anomaly score threshold using the validation set
        print("Calculating OCSVM scores on validation data...")
        # decision_function: higher score = more normal (further inside boundary)
        val_decision_scores = ocsvm_model.decision_function(X_val)
        # Invert scores so higher means more anomalous
        anomaly_scores = -val_decision_scores

        threshold = np.percentile(anomaly_scores, threshold_percentile)
        print(f"OCSVM Anomaly threshold ({threshold_percentile}th percentile of inverted scores): {threshold:.6f}")

        # Save the threshold
        joblib.dump(threshold, model_paths['ocsvm_threshold'])
        print(f"OCSVM threshold saved to {model_paths['ocsvm_threshold']}")


    print("\n--- All ML Model Training Complete ---")
    return True