import csv
import os
from datetime import datetime

FILE = "data/retrain_data.csv"

def log_for_retraining(tx, label):
    os.makedirs("data", exist_ok=True)
    exists = os.path.exists(FILE)

    with open(FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(tx.keys()) + ["isFraud"])
        if not exists:
            writer.writeheader()
        writer.writerow({**tx, "isFraud": label})
