# Data provenance and preparation

See [research](../docs/research.md) for source, terms, size and suitability assessed before download. Data and labels are excluded from Git. `results/dataset_manifest.json` is the portable record of images actually used.

For this non-commercial research project:

```powershell
python -m scripts.prepare_dataset --accept-research-terms --train 64 --val 12 --test 12
```

For interrupted downloads, append `--resume`. It refuses to overwrite a completed manifest. A source revision change requires deliberate review. Full-resolution mirror images and COCO-style xywh boxes are converted into normalized, single-class YOLO labels. Train/val/test remain separate; no random reshuffling across splits occurs.

If the mirror is unavailable, download the official archive manually through the [authors' links](https://github.com/eg4000/SKU110K_CVPR19). Extract to an ignored location containing `images/` and `annotations/annotations_train.csv`, `annotations_val.csv`, `annotations_test.csv`. Then:

```powershell
python -m scripts.prepare_dataset --official-root data/raw/SKU110K_fixed --output data/sku110k_official --accept-research-terms --train 512 --val 64 --test 64
```

Official conversion uses lexicographically sorted filenames, so it is a different subset from the mirror run. Do not compare runs as if their samples were identical. No automatic multi-gigabyte archive download is performed.

Use `data/sku110k/images/test` directly for the local benchmark; no duplicate image directory is necessary. Sample and failure-analysis images containing restricted data remain local and ignored. Public documentation includes a screenshot of the empty upload page; analyzed screenshots are available locally but excluded from Git pending any desired redistribution review.
