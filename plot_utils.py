import matplotlib.pyplot as plt
import os
from datetime import datetime


def save_plots_to_folder(folder_name="plots"):
    """Create plots folder and save current plot"""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"plot_{timestamp}.png"
    filepath = os.path.join(folder_name, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {filepath}")
    return filepath
