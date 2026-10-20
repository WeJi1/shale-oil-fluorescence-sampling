# Reproducible Analysis of Spectral Sensitivity and Spatial ROI Sampling in Shale-Oil Fluorescence Imaging

**Status: candidate research-reproducibility release; NOT public until authors approve data ownership, permissions, license, and metadata.** This is a newly defined, auditable image-analysis implementation (2026-10-19), **not recovery of the original historical algorithm**.

## Scope

- Image-level multiscale positional CV on eight archived stitched 8-bit RGB navigation maps; these are **not** original confocal single-channel detector measurements.
- Finite-design exact CV of a hotspot-fraction ROI estimator under specified stratified random placement.
- Primary image-adaptive allocation: `[8,4,12,2,12,10,11,9]` ROIs; uniform allocation `12` per map. Result: 68 versus 96 *planned* ROIs (29.17% count reduction), **not instrument time saving, not chemical measurement accuracy**.
- Six original statistical-engine tests plus eight image-processing software tests passed; conditional fixed-seed Monte Carlo results are in `data/derived/`.

## What the repository **does not** validate

- The historical spreadsheet result 69/96 (28.125%) and its lost placement/weighting settings.
- Chemical identity/abundance from colors, the universal validity of a 600 nm hydrocarbon boundary, or direct spectral–spatial uncertainty propagation.
- Prospective data acquisition, sample-level independent replicates, or generalization beyond eight maps.

## Reproduce

```
python -m pip install -r requirements.txt
python scripts/test_cv_engine.py
python scripts/test_new_image_audit.py
python scripts/final_image_audit.py --images data/raw_navigation --out results/new_run
```

**IMPORTANT:** the eight source navigation images are **intentionally omitted from this draft public package** pending documented permission from the commissioned project owner and data contributors. To reproduce outputs from image pixels, the authorized original `NAV01_...png` through `NAV08_...png` files must be placed in `data/raw_navigation/`; verify checksums against `data/metadata/Inputs_Sha256.json`. Without these images readers can inspect code, tests, parameters and deposited numerical outputs but **cannot claim complete image-to-result reproducibility**.

The numerical dataset is supplied as archived derived CSV outputs. `data/metadata/PARAMS.json` gives the exact image-based analysis design and masks; the code does not use the old spreadsheets as target values. New results can differ from historical figures. New scores and old numbers must never be mixed in a single result.

## Spectral data

`Fig2_*` and `Fig3_*` CSVs are digitized **selected-domain** spectra and descriptors from an archived workbook. The source figures and instrument files are not part of this public draft. The 500–735 nm and 500–794 nm intervals do not establish full spectrometer acquisition ranges. The fraction labels in °C are fractionation-class labels, not spectroscopy acquisition temperatures.

## Release and permissions checklist

1. Confirm ownership and redistribution rights for the commissioned sample maps and archived plots; confirm no confidential coordinates, well records, or instrument metadata are exposed.
2. Review code, authors/contributors, optional `CITATION.cff`, repository license (none granted by this candidate package), and corresponding author contact.
3. If permitted, add original images and publish a versioned GitHub release; optionally archive that exact release in Zenodo for a version-specific DOI.
4. If source images cannot be made public, keep a restricted reviewer data route and use a candid limited-availability statement. Never imply open reproduction from code alone.

## Relevant manuscript figures

- Main spectral: Fig.2 (fraction spectra), Fig.3 (operational boundary scan).
- Main spatial: Fig.4 (original image montage), NEW Fig.5 (multiscale positional CV), NEW Fig.6 (retrospective ROI allocation).
- Historical Moran prediction (formerly Fig.6) is **not** a validated predictor for the new N labels; if retained, label historical exploratory analysis in SI.

Prepared from a restricted user-provided research archive. All scientific interpretations, public releases and authorship approvals belong to the investigators.
