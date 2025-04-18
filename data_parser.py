# data_parser.py
# Functions for parsing CAN log files.

import re
import os
from can_message import CANMessage # Import from local module

def parse_can_log_line(line):
    """Parses a single line from the CAN-Intrusion Dataset .txt format."""
    match = re.match(r"Timestamp:\s*([\d.]+)\s*ID:\s*([0-9a-fA-F]+)\s*\d+\s*DLC:\s*(\d+)\s*(.*)", line)
    if match:
        try:
            timestamp = float(match.group(1))
            can_id_hex = match.group(2)
            can_id_int = int(can_id_hex, 16)
            dlc = int(match.group(3))
            data_str = match.group(4).strip()
            data_bytes = bytes.fromhex(data_str.replace(" ", ""))
            if not (0 <= dlc <= 8): return None # Invalid DLC
            return {'timestamp': timestamp, 'can_id': can_id_int, 'dlc': dlc, 'data_bytes': data_bytes}
        except Exception: # Catch broader exceptions during parsing
            return None
    return None

def load_and_parse_data(file_path, assigned_label):
    """Loads .txt data, parses it, and assigns the given label."""
    print(f"\n--- Loading Data: {file_path} (Label: {assigned_label}) ---")
    messages = []
    line_count = 0
    parse_errors = 0

    if not os.path.exists(file_path):
        print(f"ERROR: Data file not found at {file_path}.")
        return []

    try:
        with open(file_path, 'r') as f:
            for line in f:
                line_count += 1
                parsed = parse_can_log_line(line)
                if parsed:
                    messages.append(CANMessage(
                        timestamp=parsed['timestamp'],
                        can_id=parsed['can_id'],
                        dlc=parsed['dlc'],
                        data=parsed['data_bytes'],
                        label=assigned_label
                    ))
                else:
                    # Reduce verbosity of parse errors, just count them
                    parse_errors += 1
                if line_count % 100000 == 0: print(f"  Parsed {line_count} lines...")

        print(f"Read {line_count} total lines. Successfully parsed: {len(messages)}. Errors: {parse_errors}")
        if parse_errors > 0:
            print(f"Warning: Encountered {parse_errors} lines that could not be parsed.")
        print(f"--- Data Loading Complete ({len(messages)} messages) ---")
        return messages
    except Exception as e:
        print(f"ERROR: Failed to process data file {file_path}: {e}")
        return []