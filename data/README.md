# Data

This project uses **BRISC2025** (Fateh et al., 2025): 6,000 T1-weighted brain MRI slices in
4 classes, with expert tumor masks. The dataset is **not** committed to this repository
(~295 MB, ~15.6k files). Get it from the official source:
<https://www.kaggle.com/datasets/briscdataset/brisc2025>. Its license and terms of use are
listed on that page.

## Option A: automatic download (default)

If `data/brisc2025/` does not exist, the notebook downloads the dataset with `kagglehub`
(cached in `~/.cache/kagglehub`). This needs a Kaggle API token:

1. kaggle.com → *Settings* → *API* → *Create New Token*. This downloads `kaggle.json`.
2. Move it to `~/.kaggle/kaggle.json` (Windows: `C:\Users\<you>\.kaggle\kaggle.json`) and
   run `chmod 600 ~/.kaggle/kaggle.json`.

## Option B: local copy

Download the archive from the Kaggle page and extract it so the layout matches the tree
below. The folder that contains `classification_task/` must be `data/brisc2025/`. The
notebook uses this copy automatically. To use a copy somewhere else, set
`BRISC_DATA_ROOT=/path/to/brisc2025` before starting Jupyter.

To check the copy:

```bash
python scripts/check_data.py                 # file counts per folder
python scripts/check_data.py --verify-hashes # also SHA-256 of every file against manifest.csv
```

## Expected structure

```
data/brisc2025/
├── classification_task/
│   ├── train/{glioma,meningioma,no_tumor,pituitary}/   5,000 .jpg (1147 / 1329 / 1067 / 1457)
│   └── test/{glioma,meningioma,no_tumor,pituitary}/    1,000 .jpg (254 / 306 / 140 / 300)
├── segmentation_task/
│   ├── train/{images,masks}/                           3,933 image/mask pairs
│   └── test/{images,masks}/                              860 image/mask pairs
├── manifest.csv / manifest.json (+ .sha256)
└── README.md
```

Filenames follow `brisc2025_<split>_<index>_<tumor>_<view>_<sequence>.<ext>`, where
`tumor` is one of `gl` / `me` / `nt` / `pi` and `view` is `ax` / `co` / `sa`. Images are
`.jpg`. Masks have the same basename with a `.png` extension. The segmentation task
contains tumor slices only (no `nt`).

## Citation

```bibtex
@article{fateh2025brisc,
  title   = {BRISC: Annotated Dataset for Brain Tumor Segmentation and Classification with Swin-HAFNet},
  author  = {Fateh, Amirreza and Rezvani, Yasin and Moayedi, Sara and Rezvani, Sadjad and Fateh, Fatemeh and Fateh, Mansoor and Abolghasemi, Vahid},
  journal = {arXiv preprint arXiv:2506.14318},
  year    = {2025}
}
```
