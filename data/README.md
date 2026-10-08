# Data

This project uses **BRISC2025** (Fateh et al., 2025): 6,000 T1-weighted brain MRI slices in
4 classes, with expert tumor masks.

**The dataset is included** in `data/brisc2025/` (~295 MB, ~15.6k files), so the notebook
works straight after cloning. No Kaggle account is needed.

## License and attribution

BRISC2025 is published by its authors on
[Kaggle](https://www.kaggle.com/datasets/briscdataset/brisc2025) under
**[Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)**.
The copy here is the **unmodified** official release, including the authors' own
`README.md` and `manifest.csv`/`manifest.json` with SHA-256 checksums. No files were
changed, added or removed. If you use this data, please cite the authors (see below).

## Checking the copy

```bash
python scripts/check_data.py                 # file counts per folder
python scripts/check_data.py --verify-hashes # also SHA-256 of every file against manifest.csv
```

## Getting it another way

- **Kaggle:** if `data/brisc2025/` is missing, the notebook downloads the dataset with
  `kagglehub` instead. This needs an API token from kaggle.com → *Settings* → *API*, saved
  as `~/.kaggle/kaggle.json` (`chmod 600`).
- **Custom location:** set `BRISC_DATA_ROOT=/path/to/brisc2025` before starting Jupyter.
  The folder must contain `classification_task/` and `segmentation_task/` directly.

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
