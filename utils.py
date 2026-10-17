# utils.py
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from IPython.display import display
from config import SEED, PLOT_STYLE

# ---------------------------------------------------------------- 1. Logging Helpers
def banner(title, width=76):
    """Prints a formatted section header."""
    print("=" * width)
    print(title)
    print("=" * width)

def ok(msg):
    print(f"  [PASS] {msg}")

def warn(msg):
    print(f"  [WARN] {msg}")

def info(msg):
    print(f"  [INFO] {msg}")

# ---------------------------------------------------------------- 2. Reproducibility Helper
def set_seed(seed=SEED):
    """Locks all random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    ok(f"Random seed locked at {seed}")

# ---------------------------------------------------------------- 3. Styling Helper
def apply_styling():
    """Applies global Matplotlib and Pandas formatting rules."""
    plt.rcParams.update(PLOT_STYLE)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 140)
    ok("Global plot and table styling applied")

# ---------------------------------------------------------------- 4. Display & Command Sequence Helpers
def show_table(data_dict, col_name="Value", index_name="Setting"):
    """Converts a dictionary into a formatted Pandas DataFrame and displays it."""
    df = pd.DataFrame({col_name: data_dict}).rename_axis(index_name)
    display(df)
    return df

def run_section(title, data_dict=None, col_name="Value", index_name="Setting"):
    """
    Combines a section banner and table display into a single command call.
    Useful for printing summary metrics at the end of every pipeline stage.
    """
    banner(title)
    if data_dict is not None:
        return show_table(data_dict, col_name=col_name, index_name=index_name)