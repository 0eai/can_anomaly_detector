# profiler.py
# Function to learn normal profile from data.

import pandas as pd
import numpy as np
import math
import os
from data_parser import load_and_parse_data # Needs the loading function

def learn_profile_from_normal_data(file_path, config):
    """
    Analyzes NORMAL traffic file to populate relevant parts of the CONFIG dict.
    Modifies the config dictionary in-place.
    Returns True on success, False on failure.
    """
    print(f"\n--- Learning Profile from: {file_path} ---")

    # Use load_and_parse_data to get CANMessage objects
    # Assign temporary 'Normal' label, not used for profiling itself
    normal_messages = load_and_parse_data(file_path, 'Normal')

    if not normal_messages:
        print("ERROR: No valid normal data loaded for profiling.")
        return False

    # Convert messages to DataFrame for easier analysis
    try:
        data_for_df = [{
            'timestamp': msg.timestamp,
            'can_id': msg.can_id,
            'dlc': msg.dlc,
            'data_bytes': msg.data # Keep as bytes for non-zero count
        } for msg in normal_messages]

        df = pd.DataFrame(data_for_df)

        if df.empty:
            print("ERROR: DataFrame created from normal messages is empty.")
            return False

        # --- Perform analysis and update config (similar logic as before) ---

        # 1. Learn Known IDs
        known_ids = set(df['can_id'].unique())
        config['static_checks']['known_ids'] = known_ids
        print(f"Learned {len(known_ids)} unique CAN IDs.")

        # 2. Learn Expected DLCs
        expected_dlcs = {}
        for can_id in known_ids:
            # Ensure the filtered DataFrame is not empty before accessing mode
            id_specific_dlc = df.loc[df['can_id'] == can_id, 'dlc']
            if not id_specific_dlc.empty:
                mode_dlc = id_specific_dlc.mode()
                if not mode_dlc.empty:
                    expected_dlcs[can_id] = int(mode_dlc[0])
        config['static_checks']['expected_dlc'] = expected_dlcs
        print(f"Learned expected DLCs for {len(expected_dlcs)} IDs.")

        # 3. Learn Timing Parameters & Imputation Value
        timing_stats = {}
        df = df.sort_values(by=['can_id', 'timestamp'])
        df['time_delta'] = df.groupby('can_id')['timestamp'].diff()
        min_perc, max_perc = config['learning']['timing_percentiles']
        all_valid_deltas = df['time_delta'].dropna()

        if not all_valid_deltas.empty:
            # Calculate median only if there are valid deltas
            median_delta = all_valid_deltas.median()
            config['feature_config']['impute_timing_value'] = median_delta
            print(f"Learned timing imputation value: {median_delta:.6f}s")
        else:
            print("Warning: No valid time deltas found. Using default imputation value.")
            # Keep default from config

        for can_id in known_ids:
            deltas = df[df['can_id'] == can_id]['time_delta'].dropna()
            if len(deltas) > 1: # Need at least 2 messages for percentiles
                min_interval = max(1e-9, np.percentile(deltas, min_perc))
                max_interval = np.percentile(deltas, max_perc)
                if max_interval <= min_interval: max_interval = min_interval + 1e-9
                timing_stats[can_id] = (min_interval, max_interval)
        config['feature_config']['normal_inter_arrival'] = timing_stats
        print(f"Learned timing stats for {len(timing_stats)} IDs.")

        # 4. Learn Payload Parameters
        payload_stats = {}
        df['non_zero_count'] = df['data_bytes'].apply(lambda b: sum(1 for x in b if x != 0))
        min_p, max_p = config['learning']['payload_percentiles']
        for can_id in known_ids:
            counts = df[df['can_id'] == can_id]['non_zero_count'].dropna()
            if len(counts) > 0:
                min_count = np.percentile(counts, min_p)
                max_count = np.percentile(counts, max_p)
                # Handle cases where min/max might be very close or equal
                min_c_floor = math.floor(min_count)
                max_c_ceil = math.ceil(max_count)
                if min_c_floor > max_c_ceil: # Ensure min <= max
                    max_c_ceil = min_c_floor
                payload_stats[can_id] = (min_c_floor, max_c_ceil)
        config['feature_config']['normal_nonzero_bytes'] = payload_stats
        print(f"Learned payload stats for {len(payload_stats)} IDs.")

        print("--- Profile Learning Complete ---")
        return True

    except Exception as e:
        print(f"ERROR during profile learning: {e}")
        import traceback
        traceback.print_exc()
        return False