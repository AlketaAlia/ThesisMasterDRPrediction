1.po# Legacy training scripts

These five near-identical scripts trained one architecture each and were the
basis of the original thesis results. They are kept here for reference and
reproducibility but are **no longer the recommended entry point**.

Use `../train.py` instead, which:

- Stratifies the train/val/test split (the originals didn't)
- Keeps the test set un-augmented (the originals augmented it — a real bug)
- Adds EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint(save_best_only=True)
- Applies architecture-specific `preprocess_input` consistently
- Supports class weighting and optional fine-tuning of the top base layers
- Replaces the five scripts with a single CLI flag (`--arch`)
