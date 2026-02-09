# Fix NumPy Compatibility Issue

## Problem
You're getting errors because NumPy 2.x is installed, but YOLOv5 dependencies (torchvision, matplotlib) require NumPy 1.x.

## Solution

Run these commands in your terminal (make sure your virtual environment is activated):

```bash
# Activate virtual environment
source venv/bin/activate

# Downgrade NumPy to version < 2.0
pip install "numpy<2.0"

# Verify the version
python -c "import numpy; print('NumPy version:', numpy.__version__)"
```

You should see NumPy version 1.x (like 1.24.3 or similar).

## Then Run Training

**Important:** Use `python` not `ipython`:

```bash
python train_model.py --data custom_logo.yaml --batch 16 --epochs 200
```

## Why This Happened

- NumPy 2.0+ introduced breaking changes
- torchvision and matplotlib were compiled with NumPy 1.x
- They need to be rebuilt or NumPy needs to be downgraded

## Alternative: Reinstall Everything

If downgrading NumPy causes other issues, you can reinstall:

```bash
pip uninstall numpy torch torchvision matplotlib
pip install "numpy<2.0"
pip install torch torchvision matplotlib
```
