# Week3/run_exp2.py
import sys, os

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath(os.path.join(".", "Week2")))

from Week2.main import run_finetune

# Experiment 2
run_finetune(
    unfreeze_last_n=50,
    lr=1e-4,
    epochs=12,
    batch_size=32
)
