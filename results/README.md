# Results

Reference metrics from the executed run of
[`../notebooks/brain_tumor_segmentation_classification.ipynb`](../notebooks/brain_tumor_segmentation_classification.ipynb)
(RTX 4080 SUPER, CUDA 12.1, PyTorch 2.5.1). They were copied from the notebook's saved cell
outputs, so they can be read without opening the notebook.

| File | Contents |
|---|---|
| `segmentation_results.csv` | U-Net segmentation: Dice, mIoU, pixel accuracy (train/val/test) |
| `attention_unet_results.csv` | Attention U-Net segmentation: Dice, mIoU, pixel accuracy |
| `classification_results.csv` | Classifier on U-Net encoder features: accuracy and macro precision/recall/F1 |
| `attention_classifier_results.csv` | Classifier on Attention U-Net encoder features: accuracy and macro precision/recall/F1 |

A new run writes the same files to `outputs/`. To compare them:

```bash
python scripts/compare_results.py
```

The notebook writes `attention_classifier_results.csv` with lowercase column names
(`acc`, `precision`, ...) and capitalised split names. The copy here uses the same format as
the other files. `compare_results.py` handles both formats.

Model checkpoints (`*.pth`: ~375 MB per segmenter including optimizer state, ~14 MB per
classifier) are **not** committed. Running the notebook
recreates them in `outputs/`.
