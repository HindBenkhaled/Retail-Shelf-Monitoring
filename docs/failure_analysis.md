# Qualitative gap failure analysis

Reviewed 6 held-out SKU-110K test images with side-by-side original, raw classical, and fused overlays. The local artifacts are under `results/failure_analysis/` and are intentionally ignored because the dataset terms are academic/non-commercial and redistribution of mirror images was not independently cleared. No gap ground truth is available, so this review does not produce an accuracy score.

## Observations

- Large plain areas outside the intended shelf region are the strongest false-positive source. Floors, ceiling panels, wall signs, end-cap side panels and blank fascia frequently have low Laplacian energy. Cropping to the shelf is important because the current denominator is the full processed image.
- Uniform or low-contrast product faces can be marked as empty. This is visible on broad beige, dark and single-color packages.
- Dark shelf recesses and shadows sometimes remain as candidate gaps. Reflections usually add edges and can instead hide a visually empty region.
- Shelf rails and price-label strips split regions into smaller components. Some plausible narrow gaps disappear after the minimum width/height/area filter.
- The product detector suppresses pixels where a detected box overlaps a raw gap, but its recall on this small CPU-trained subset is 0.406. Many products remain unboxed, especially small or distant facings, so fusion cannot correct every product-shaped false gap.
- Product detections can cover signage or packaging groups imperfectly. A false-positive box could suppress a genuine gap. Fusion is therefore evidence combination, not proof that a remaining region is empty.
- One product box can divide a connected candidate into multiple components, so fusion can reduce area while occasionally increasing gap count. Count alone is not a quality measure.

## Representative local records

| Test image | Products detected | Raw candidates | Fused candidates | Fused estimated area |
|---|---:|---:|---:|---:|
| `test_00000.jpg` | 71 | 27 | 22 | 18.9396% |
| `test_00001.jpg` | 39 | 27 | 26 | 4.4711% |
| `test_00002.jpg` | 61 | 22 | 18 | 7.7164% |
| `test_00003.jpg` | 54 | 8 | 7 | 0.6165% |
| `test_00004.jpg` | 101 | 23 | 21 | 14.8188% |
| `test_00005.jpg` | 57 | 10 | 9 | 4.4169% |

These values describe algorithm output on selected images. They are not correctness labels or calibrated shelf-capacity measurements.

## Next evaluation step

Create a separately licensed shelf-gap segmentation set with image-level grouping by store/scene and independent polygon annotations. Freeze the classical thresholds before scoring pixel IoU and region precision/recall on the test set. Until that exists, report product localization metrics quantitatively and gap behavior qualitatively.
