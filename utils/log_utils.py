import os
import datetime
import json

class Logger:
    def __init__(self, log_dir, log_name="log.txt"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, log_name)
        self._write_header()

    def _write_header(self):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"Logging started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*50 + "\n")

    def write(self, message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}\n"
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(log_msg)

    def save_dict(self, data, save_path, file_name):
        save_full_path = os.path.join(save_path, file_name)
        with open(save_full_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        self.write(f"Dictionary saved to: {save_full_path}")

    def save_metrics(self, metrics, save_path, file_name="metrics.csv"):
        save_full_path = os.path.join(save_path, file_name)
        header = ",".join(metrics.keys()) + "\n"
        values = ",".join([f"{v:.4f}" for v in metrics.values()]) + "\n"
        
        if not os.path.exists(save_full_path):
            with open(save_full_path, "w", encoding="utf-8") as f:
                f.write(header)
                f.write(values)
        else:
            with open(save_full_path, "a", encoding="utf-8") as f:
                f.write(values)
        self.write(f"Metrics saved to: {save_full_path}")