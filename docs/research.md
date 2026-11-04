# Research and design decisions

Research performed before implementation, 13 October 2026. Repository descriptions are claims by their authors, not independently verified performance evidence. No third-party project code was copied.

## Public implementations inspected

| Source | Useful pattern | Decision here |
|---|---|---|
| [SKU110K_CVPR19](https://github.com/eg4000/SKU110K_CVPR19) | Dense product localization, explicit annotations and separate evaluation | Use one product class and preserve dataset splits; the original RetinaNet model is not YOLOv8 |
| [Shelf-Sight](https://github.com/jackson-marcus/Shelf-Sight) | Inspectable classical stages and separated serving | Expose raw and fused masks; use the requested Laplacian method instead of its color-based pipeline |
| [Retail-Shelf-Monitoring](https://github.com/Alijanloo/Retail-Shelf-Monitoring) | Separate detection, recognition and operational components | Keep a small modular image analyzer; exclude tracking, SKU recognition and planogram claims. Its advertised YOLOv12 is not used |
| [Grocery-Product-Detection](https://github.com/sayakpaul/Grocery-Product-Detection) | Dataset conversion, training and inference workflows | Provide separate reproducible commands and an isolated test split |

## Dataset assessment before download

| Dataset | Source / usage terms | Approximate size and classes | Practicality |
|---|---|---|---|
| SKU-110K | [Original authors](https://github.com/eg4000/SKU110K_CVPR19): academic and non-commercial use only, exclusive use by recipient | [Ultralytics dataset documentation](https://docs.ultralytics.com/datasets/detect/sku-110k): 11,743 images, one generic object class; train 8,219 / validation 588 / test 2,936; approximately 13.6 GB | Good task match; full download/training is excessive for a CPU development run |
| Grocery Dataset (Varol & Kuzu) | [Official repository](https://github.com/gulvarol/grocerydataset): research only, non-commercial | 354 shelf images; 10 product categories; approximately 3 GB across two archives, including product crops | Smaller but still a large archive; related shelf views require careful split grouping |
| Supervisely Grocery Store Shelves | [Publisher repository](https://github.com/supervisely-ecosystem/grocery-store-shelves) | 37 shelf images; one product box class; archive size not verified | Not selected: a clear dataset usage license was not established during inspection |

## Selected acquisition

SKU-110K, through the [benjamintli mirror](https://huggingface.co/datasets/benjamintli/sku110k). The mirror card explicitly attributes the original authors and describes its conversion. It does not replace the original usage terms. Revision: `41ffe587d4e568204bbf752ba8f18ea21e3d7b09`.

Use the first 64 train, 12 validation, and 12 test records in mirror order. This is a development/feasibility subset, not a random representative sample. Downloads use the public dataset viewer rows API to avoid multi-gigabyte parquet shards. Before downloading images the script requires acknowledgment of the documented research terms. It checks revision, class IDs, dimensions, decodability, bounding boxes and duplicate image hashes. A manifest records every acquired image, dimensions, box counts, bytes and hash; exact byte totals are measured only after acquisition.

The mirror is a third-party conversion; the original data were not downloaded for a byte-for-byte cross-check. Split labels and box-format consistency are checked, but preservation of every original record is not independently certified. The test split is not used to select epochs or tune thresholds. Near-duplicate scenes may remain even when byte hashes differ.

No verified gap segmentation ground truth was acquired. Product AP does not measure gap quality. Gap analysis is qualitative and explicitly heuristic.

## API/model references

- [Official YOLOv8 model documentation](https://docs.ultralytics.com/models/yolov8/): explicitly select `yolov8n.pt`; do not inherit newer default families.
- [Training arguments](https://docs.ultralytics.com/modes/train/) and [validation metrics](https://docs.ultralytics.com/modes/val/).
- [Ultralytics licensing](https://www.ultralytics.com/license): AGPL-3.0 / enterprise options. Third-party data terms remain separate from software licensing.

## Deliberate scope

The service processes uploaded still images. It does not infer stock counts behind visible products, identify SKUs, detect temporal events, or promise camera-stream frame rates. The project title describes the low-latency monitoring use case; performance claims refer only to measured warm image processing on the stated hardware.
