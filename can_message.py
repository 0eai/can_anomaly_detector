# can_message.py
# Defines the CANMessage data structure.

class CANMessage:
    """Simple representation of a CAN message."""
    def __init__(self, timestamp, can_id, dlc, data, label=None):
        self.timestamp = float(timestamp)
        self.can_id = int(can_id)
        self.dlc = int(dlc)
        self.data = bytes(data) if isinstance(data, bytes) else bytes(data)
        # Pad or truncate data based on DLC
        if len(self.data) < self.dlc:
            self.data += b'\x00' * (self.dlc - len(self.data))
        elif len(self.data) > self.dlc:
            self.data = self.data[:self.dlc]
        self.label = label # Store the ground truth label

    def __repr__(self):
        label_str = f", Lbl={self.label}" if self.label else ""
        return f"CAN(ts={self.timestamp:.6f}, id=0x{self.can_id:X}, dlc={self.dlc}, data={self.data.hex()}{label_str})"