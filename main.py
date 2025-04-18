# main.py
# Main script to run the CAN anomaly detection simulation.

import time
import os
import random
import numpy as np # Used for seeding

# Import from local modules
import config as cfg # Use alias for clarity
from profiler import learn_profile_from_normal_data
from training import train_ml_models
from detector import HybridAnomalyDetector
from data_parser import load_and_parse_data
from evaluation import evaluate_results

def main():
    """Main execution function."""

    # Load configuration
    config = cfg.get_config()

    # Set random seed for reproducibility if specified
    seed = config.get('training_options', {}).get('random_seed', None)
    if seed is not None:
        print(f"Setting random seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        # Set TensorFlow seed if TensorFlow is used
        try:
            import tensorflow as tf
            tf.random.set_seed(seed) # <<< ERROR HAPPENS HERE
            # Optional: Configure for deterministic operations if possible
            # os.environ['TF_DETERMINISTIC_OPS'] = '1'
            # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        except ImportError:
            pass # TF not installed or used
        except AttributeError: # Add this to catch the specific error gracefully
            # Fallback for TensorFlow 1.x
            try:
                print("INFO: Using tf.set_random_seed() for TensorFlow 1.x compatibility.")
                tf.set_random_seed(seed) # <<< USE THIS INSTEAD FOR TF 1.x
            except AttributeError:
                print("Warning: Could not set TensorFlow random seed (tf.set_random_seed not found).")
            except NameError: # If tf wasn't imported successfully before the first AttributeError
                pass
        except Exception as e: # Catch any other unexpected errors during seeding
            print(f"Warning: An error occurred while setting TensorFlow seed: {e}")

    # Define file paths from config
    data_paths = config['data_paths']
    normal_file_path = os.path.join(data_paths['data_dir'], data_paths['normal_filename'])
    attack_filenames = data_paths['attack_filenames']

    try:
        # === STAGE 1: Learn Static Profile & Feature Stats ===
        profile_learned = learn_profile_from_normal_data(normal_file_path, config)
        if not profile_learned:
            raise RuntimeError("Failed to learn profile from normal data. Aborting.")

        # === STAGE 2: Train ML Models (Offline Step) ===
        # This trains models based on normal data and saves them
        models_trained = train_ml_models(normal_file_path, config)
        if not models_trained:
            raise RuntimeError("Failed to train ML models. Aborting.")

        # === STAGE 3: Initialize Detector (Loads trained models) ===
        print("\n--- Initializing Hybrid Detector with Trained Models ---")
        # Detector loads models using paths defined in config
        detector = HybridAnomalyDetector(config)

        # === STAGE 4: Load Combined Test Data ===
        print("\n--- Loading Combined Test Data for Inference ---")
        combined_test_messages = []
        message_count_details = {}

        # Load Normal Data portion for testing
        print(f"Loading Normal test portion: {data_paths['normal_filename']}")
        normal_test_messages = load_and_parse_data(normal_file_path, 'Normal')
        if normal_test_messages:
            # Optional: Limit number of normal messages for faster testing/balance
            # normal_limit = 50000
            # combined_test_messages.extend(random.sample(normal_test_messages, min(normal_limit, len(normal_test_messages))))
            # message_count_details['Normal'] = len(combined_test_messages)
            combined_test_messages.extend(normal_test_messages) # Use all
            message_count_details['Normal'] = len(normal_test_messages)

        else:
            print(f"Warning: Could not load normal data from {normal_file_path}")

        # Load Attack Data portions
        for attack_filename in attack_filenames:
            attack_file_path = os.path.join(data_paths['data_dir'], attack_filename)
            print(f"Loading Attack test portion: {attack_filename}")
            attack_messages = load_and_parse_data(attack_file_path, 'Anomaly')
            if attack_messages:
                # Optional: Limit attack messages
                # attack_limit = 20000
                # combined_test_messages.extend(random.sample(attack_messages, min(attack_limit, len(attack_messages))))
                # message_count_details[attack_filename] = len(combined_test_messages) - message_count_details.get('Normal', 0) - sum(v for k,v in message_count_details.items() if k != 'Normal')
                combined_test_messages.extend(attack_messages) # Use all
                message_count_details[attack_filename] = len(attack_messages)

            else:
                print(f"Warning: Could not load attack data from {attack_file_path}")

        # Shuffle the combined dataset
        print("Shuffling combined data...")
        random.shuffle(combined_test_messages)
        print(f"--- Combined Test Data Loaded ---")
        print(f"Message Counts: {message_count_details}")
        print(f"Total Messages for Inference: {len(combined_test_messages)}")

        # === STAGE 5: Process Test Data (Inference) ===
        if combined_test_messages:
            print(f"\n--- Performing Inference ({len(combined_test_messages)} messages) ---")
            start_time = time.time()
            # Reset detector state before inference run
            detector.reset_state()

            for i, msg in enumerate(combined_test_messages):
                detector.process_message(msg)
                if (i + 1) % 100000 == 0: # Progress indicator
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    print(f"  Processed {i+1}/{len(combined_test_messages)} messages... ({rate:.0f} msg/s)")

            end_time = time.time()
            total_processing_time = end_time - start_time
            num_processed_messages = len(combined_test_messages)

            print(f"--- Inference Complete ---")
            print(f"Total processing time: {total_processing_time:.2f} seconds for {num_processed_messages} messages")

            # === STAGE 6: Evaluate Performance ===
            results = detector.get_results()
            evaluate_results(results, total_time=total_processing_time, num_messages=num_processed_messages)
        else:
            print("No combined test messages loaded, skipping inference and evaluation.")

    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: Data directory or essential file not found.")
        print(f"Please ensure DATA_DIR in config.py is correct ('{cfg.DATA_DIR}') and dataset files exist.")
        print(f"Details: {e}")
    except RuntimeError as e:
        print(f"\nFATAL ERROR: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during simulation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()