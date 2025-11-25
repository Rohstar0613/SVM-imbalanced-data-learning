import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

def save_png(fig, prefix="visual", folder="data/results"):
    """PNG 이미지로 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.png"
    filepath = os.path.join(folder, filename)

    os.makedirs(folder, exist_ok=True)  # 폴더 없으면 생성
    plt.savefig(filepath, dpi=300)

    print(f"📁 PNG 저장됨: {filepath}")


def save_txt(report, prefix="report", folder="data/results"):
    """텍스트 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.txt"
    filepath = os.path.join(folder, filename)

    os.makedirs(folder, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"📁 TXT 저장됨: {filepath}")


def save_csv(df, prefix="predictions", folder="data/results"):
    """DataFrame을 CSV 형식으로 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.csv"
    filepath = os.path.join(folder, filename)

    os.makedirs(folder, exist_ok=True)
    df.to_csv(filepath, index=False)

    print(f"📁 CSV 저장됨: {filepath}")


