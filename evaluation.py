# evaluation.py
# Contains the evaluation logic.

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def evaluate_results(results, total_time=None, num_messages=None):
    """
    Calculates and prints performance metrics from a list of result dicts.

    Args:
        results (list): List of dictionaries, each containing at least
                        'true_label' ('Normal'/'Anomaly') and
                        'predicted_label' ('Normal'/'Anomaly').
        total_time (float, optional): Total processing time in seconds.
        num_messages (int, optional): Total number of messages processed.
    """
    if not results:
        print("\nNo results to evaluate.")
        return

    print("\n--- Evaluation ---")
    df_results = pd.DataFrame(results)

    if 'true_label' not in df_results.columns or 'predicted_label' not in df_results.columns:
        print("ERROR: Results DataFrame missing 'true_label' or 'predicted_label' columns.")
        return

    # Convert 'Normal'/'Anomaly' labels to binary (0/1)
    try:
        true_labels_binary = df_results['true_label'].apply(lambda x: 0 if x == 'Normal' else 1)
        pred_labels_binary = df_results['predicted_label'].apply(lambda x: 0 if x == 'Normal' else 1)
    except Exception as e:
        print(f"Error converting labels to binary: {e}")
        print("Sample true labels:", df_results['true_label'].unique())
        print("Sample predicted labels:", df_results['predicted_label'].unique())
        return

    # --- Calculate Core Metrics ---
    try:
        # Check for presence of both classes
        unique_true = np.unique(true_labels_binary)
        unique_pred = np.unique(pred_labels_binary)
        if len(unique_true) < 2: print("Warning: Only one class present in true labels. Metrics might be misleading.")
        if len(unique_pred) < 2: print("Warning: Detector only predicted one class.")


        cm = confusion_matrix(true_labels_binary, pred_labels_binary, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        accuracy = accuracy_score(true_labels_binary, pred_labels_binary)
        # Use zero_division=0 to handle cases where denominator is zero
        precision = precision_score(true_labels_binary, pred_labels_binary, zero_division=0, pos_label=1)
        recall = recall_score(true_labels_binary, pred_labels_binary, zero_division=0, pos_label=1) # Same as DR
        f1 = f1_score(true_labels_binary, pred_labels_binary, zero_division=0, pos_label=1)

        detection_rate = recall # DR is Recall

        # FPR = FP / (FP + TN)
        denominator_fpr = fp + tn
        false_positive_rate = fp / denominator_fpr if denominator_fpr > 0 else 0.0

        print("Confusion Matrix (Rows: Actual, Columns: Predicted):")
        print(f"          Normal | Anomaly")
        print(f"Normal    {tn:<7} | {fp:<7}")
        print(f"Anomaly   {fn:<7} | {tp:<7}")

        print(f"\nMetrics (Positive Class: Anomaly):")
        print(f" Accuracy:          {accuracy:.4f}")
        print(f" Precision:         {precision:.4f} (TP / (TP + FP))")
        print(f" Recall (DR, TPR):  {recall:.4f} (TP / (TP + FN))")
        print(f" F1-Score:          {f1:.4f}")
        print(f" Detection Rate:    {detection_rate:.4f} (Same as Recall)")
        print(f" False Positive Rate: {false_positive_rate:.4f} (FP / (TN + FP))") # Corrected label

    except ValueError as e:
        print(f"Warning: Could not calculate full metrics ({e}). Check label distribution.")
        print("True Label Distribution:\n", true_labels_binary.value_counts())
        print("Predicted Label Distribution:\n", pred_labels_binary.value_counts())
    except Exception as e:
        print(f"An unexpected error occurred during metrics calculation: {e}")


    # --- Calculate and Print Processing Latency ---
    if total_time is not None and num_messages is not None and num_messages > 0:
        avg_latency_ms = (total_time / num_messages) * 1000
        print(f"\nProcessing Performance:")
        print(f" Total Messages Processed: {num_messages}")
        print(f" Total Processing Time:  {total_time:.2f} seconds")
        print(f" Average Latency:        {avg_latency_ms:.4f} ms/message")
    else:
        print("\nProcessing Latency information not available.")

    # --- Analyze Misclassifications ---
    try:
        fp_df = df_results[(true_labels_binary == 0) & (pred_labels_binary == 1)]
        fn_df = df_results[(true_labels_binary == 1) & (pred_labels_binary == 0)]
        print(f"\nFalse Positives (Normal predicted as Anomaly): {len(fp_df)}")
        # Optionally print details for a small number of FPs
        if not fp_df.empty and len(fp_df) < 15:
            # Ensure 'reason' column exists before trying to access it
            if 'reason' in fp_df.columns:
                print(fp_df[['timestamp', 'can_id', 'reason']].head())
            else:
                print(fp_df[['timestamp', 'can_id']].head())


        print(f"False Negatives (Anomaly predicted as Normal): {len(fn_df)}")
        # Optionally print details for a small number of FNs
        if not fn_df.empty and len(fn_df) < 15:
            print(fn_df[['timestamp', 'can_id', 'predicted_label']].head())
    except KeyError as e:
        print(f"Warning: Column missing for misclassification analysis: {e}")
    except Exception as e:
        print(f"Error during misclassification analysis: {e}")