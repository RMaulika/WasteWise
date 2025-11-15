# Week3/run_exp1.py
import sys, os

# 1) ensure project root is on path
sys.path.append(os.path.abspath("."))

# 2) ensure Week2/ is on path so "src" inside Week2 can be imported as top-level
sys.path.append(os.path.abspath(os.path.join(".", "Week2")))

# Now import the runner from Week2
from Week2.main import run_finetune

# Experiment 1
run_finetune(
    unfreeze_last_n=20,
    lr=5e-5,
    epochs=12,
    batch_size=32
)
