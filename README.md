# Training-Data Format Bias and Parameter-Efficient Fine-Tuning for AI-Generated Image Detection

Code, frozen protocols, trained adapters and heads, and evaluation results for the paper

> J. Wie, N. Salim, and R. R. Andisman, "Isolating the Effect of Training-Data Format Bias on Parameter-Efficient
> Fine-Tuning for AI-Generated Image Detection," submitted to the International Conference on Informatics, Multimedia,
> Cyber and Information Systems (ICIMCIS), 2026.

DOI of this record: [10.5281/zenodo.23538937](https://doi.org/10.5281/zenodo.23538937)

## What the study does

A 2 × 2 design crosses the **adaptation strategy** (linear probe, LP, vs. low-rank adaptation, LoRA) with the
**training-data encoding** (D-bias vs. D-align), with the backbone, image content and labels held fixed:

- **D-bias**: GenImage Stable Diffusion v1.4 images with JPEG real and PNG synthetic images (the documented format bias).
- **D-align**: the same images, with every synthetic image JPEG-encoded once using the quantization tables, chroma
  subsampling and JFIF header of a randomly paired real image of the same class.

The confirmatory backbone is DINOv2 ViT-L/14; CLIP ViT-L/14 (the backbone of UnivFD) is an exploratory replication.
Models are evaluated on Synthbuster with RAISE-1k photographs, and on the WildRF test split as a robustness check.
The confirmatory contrast is the interaction I = Δ(LoRA) − Δ(LP) in balanced accuracy, with five seeds per cell.

| Backbone | Test set | Δ(LP) | Δ(LoRA) | I (95% t-interval) | R |
|---|---|---|---|---|---|
| DINOv2 ViT-L/14 | Synthbuster | +0.004 | +0.174 | **+0.171** [0.100, 0.241], supported | −0.216 |
| DINOv2 ViT-L/14 | WildRF | −0.007 | +0.110 | **+0.117** [0.095, 0.138], supported | −0.225 |
| CLIP ViT-L/14 | Synthbuster | +0.074 | +0.167 | +0.093 [−0.011, 0.197], inconclusive | −0.026 |
| CLIP ViT-L/14 | WildRF | −0.089 | +0.012 | +0.101 [0.063, 0.139], supported | −0.102 |

## Contents

| Path | Contents |
|---|---|
| `build_dataset_v2.ipynb` | Builds D-bias and D-align from GenImage SD v1.4: selection, near-duplicate removal, 336 × 336 crops (lossless DCT-domain crops for real images, off-grid offsets for synthetic images), JPEG encoding with paired quantization tables, and the two encoding audits (metadata classifier, grid-shift test; Table III) |
| `train-v7.ipynb` | DINOv2 ViT-L/14: LP and LoRA training, learning-rate grid, selection and five seeds under the frozen protocol (fingerprint `455a8c4a00baa998`, frozen 15 October 2026) |
| `train_clip.ipynb` | The same protocol on CLIP ViT-L/14 (fingerprint `d42b95eafc43c153`, frozen 16 October 2026) |
| `prep_synthbuster.ipynb` | Downloads Synthbuster from Zenodo and the RAISE-1k TIFFs listed in the RAISE CSV, makes 336 × 336 center crops without resizing (shorter side < 336 px excluded) and the JPEG copies at q = 50–95 used by the recompression diagnostic |
| `eval_kaggle.ipynb` | Evaluation of the DINOv2 runs on Synthbuster, including the recompression diagnostic, the paired bootstrap and the learning-rate sensitivity analysis |
| `eval_kaggle_clip.ipynb` | Evaluation of the CLIP runs on Synthbuster, and of the released UnivFD detector |
| `eval_wildrf.ipynb` | Evaluation of the DINOv2 and CLIP runs and of UnivFD on the WildRF test split |
| `identifier/` | `manifest.csv` (one row per training/validation image: GenImage source file, split, class, crop offset, paired real image, SHA-256 of the D-bias and D-align files), `d_bias.csv`, `d_align.csv`, selection and audit files |
| `protocol/` | `protocol_frozen*.json` (frozen protocol, date and fingerprint) and `selection_*.json` (selected learning rates) for both backbones |
| `Adapter and heads/` | Trained models: for each LoRA run the PEFT adapter (`adapter/`) and linear head (`head.pt`); for each LP run the linear head; `metrics.json` and validation predictions (`val_preds.npz`) per run |
| `prep_synthbuster/` | `manifest.csv` and `exclusions.csv` of the Synthbuster + RAISE-1k evaluation set (which images were kept or excluded, and why) |
| `results/` | Evaluation outputs: `summary.txt` (start here), `metrics_runs.csv`, `metrics_per_generator.csv`, `contrasts.csv`, `bootstrap.csv`, `sensitivity_I.csv`, `recompression.csv` (DINOv2 on Synthbuster), `univfd_repro.csv`, and per-image posteriors in `preds/*.npz`; WildRF results with its `manifest.csv` and `exclusions.csv` |

Run names follow `<backbone>_<strategy>_<version>_lr<lr>_s<seed>`, e.g. `dinov2_lora_d_align_lr0.0002_s3`.

## Data sources (not redistributed)

The images are not included because they belong to third-party datasets with their own terms of use. Obtain them from
the original sources:

- **GenImage** (Stable Diffusion v1.4 subset): Zhu et al., NeurIPS 2023. Its real images come from ImageNet.
- **Synthbuster**: Bammey, IEEE OJSP 2024, [10.5281/zenodo.10066460](https://doi.org/10.5281/zenodo.10066460).
- **RAISE-1k**: Dang-Nguyen et al., ACM MMSys 2015, available from its authors after registration.
- **WildRF**: Cavia et al., arXiv:2406.09398, linked from https://github.com/barcavia/RealTime-DeepfakeDetection-in-the-RealWorld.

## Reproducing the study

All notebooks ran on Kaggle (2 × NVIDIA T4, 16 GB; PyTorch 2.11, Transformers 5.16, PEFT 0.20). Run them in this order,
adding the output of each step as input to the next:

1. `build_dataset_v2.ipynb` with GenImage SD v1.4 attached. The rebuilt files can be checked against the SHA-256 values
   in `identifier/manifest.csv`.
2. `train-v7.ipynb` (DINOv2) and `train_clip.ipynb` (CLIP) with the build output attached. Each refuses to run if the
   protocol or the training code differs from its frozen fingerprint.
3. `prep_synthbuster.ipynb` (CPU) with the RAISE-1k CSV attached.
4. `eval_kaggle.ipynb` and `eval_kaggle_clip.ipynb` with the matching training output and the Synthbuster output attached.
5. `eval_wildrf.ipynb` with both training outputs and WildRF attached.

Each evaluation notebook writes `eval_local.py`, which can also be run on a local GPU (see its header).

To re-analyse the paper's numbers without any GPU, use the per-image posteriors in `results/.../preds/` with the
labels in `preds/_index.npz`.

## Licenses

- Code (notebooks and `eval_local.py`): MIT License.
- Trained adapters, heads, protocol files, identifier lists and results: CC BY 4.0. The adapters must be used with the
  original backbone weights, which keep their own licenses (DINOv2: Apache 2.0; OpenAI CLIP: MIT).
- The source datasets keep their own terms of use, and nothing here grants rights to them.
