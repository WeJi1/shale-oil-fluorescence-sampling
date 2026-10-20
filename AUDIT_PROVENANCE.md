# Provenance and limitations

The new algorithm was created during the 2026-10-19 audit. Unlike the unrecovered historical spatial scripts, it defines the annotation mask, top-5%-by-rank hotspot threshold, ROI geometry, strata, allowed overlap, equal ROI estimator and finite-design exact CV. This retrospective analysis is a computation on fixed source images, not independent physical sample measurements. See `data/metadata/PARAMS.json`, `scripts/final_image_audit.py`, and test scripts.

The eight normalized images must match archived source checksums. Primary source is a commissioned 2023 project report's imaging workflow; only a transcription of the report was located. Source instrument fluorescence exports, physical map calibrations and complete historical ROI-placement logs remain missing.

The selected digital spectral intervals 500–735 nm (fraction series) and 500–794 nm (two-well boundary sweep) are not authenticated full detector spectral ranges. Both spectral series are digitized from archived plots and are *unpaired* with the maps and available GC charts. No chemical composition calibration, spectral-spatial coupling, independent reference material or measured field acquisition-time validation is provided.

The historical 69/96 ROI result and the new 68/96 ROI result represent distinct analysis definitions. The archived map-level Moran prediction was trained against the OLD N labels and must not be interpreted as validation of the NEW allocation outputs.
