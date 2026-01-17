import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Ubah ke path results.csv kamu
csv_path = Path("runs/detect/train/results.csv")

df = pd.read_csv(csv_path)

# rapikan nama kolom (kadang ada spasi di header)
df.columns = [c.strip() for c in df.columns]

# --- Plot loss (train/val) ---
plt.figure()
for col in ["train/box_loss", "train/cls_loss", "train/dfl_loss",
            "val/box_loss", "val/cls_loss", "val/dfl_loss"]:
    if col in df.columns:
        plt.plot(df["epoch"], df[col], label=col)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title("Loss per epoch")
plt.show()

# --- Plot metrics utama ---
plt.figure()
for col in ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
    if col in df.columns:
        plt.plot(df["epoch"], df[col], label=col)
plt.xlabel("epoch")
plt.ylabel("score")
plt.legend()
plt.title("Detection metrics per epoch")
plt.show()
