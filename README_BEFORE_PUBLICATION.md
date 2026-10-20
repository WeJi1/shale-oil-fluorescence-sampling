# Pre-publication release gate — review before changing this repository to Public

This ZIP is a **candidate**, not evidence of an already created GitHub repository, uploaded dataset, or registered DOI.

## Included
- Python programs, parameter and SHA-256 metadata, unit checks and derived values for NEW68 image sampling and fair-budget/holdout comparisons.
- Digitized experimental spectral data and derived numerical tables used for manuscript Figures 2, 3, 5 and 6, plus Supplementary Fig. S1 numerical values.

## Not included
- Eight original THUNDER navigation images, raw Leica SP8 files, commissioned 2023 laboratory report, identifiable unpublished sample records, sample-depth ledgers or other proprietary material.

## Intellectual property / author approval
Obtain confirmation from the project-data owner and all relevant authors **before** posting any derivative dataset, output tables or scripts publicly. The lack of raw images is intentional and means image-to-result regeneration cannot run from this package alone; reviewers can inspect tables and mathematical logic, and authorized users may add the source PNGs locally as documented by the image scripts.

## Analytical meaning
The hotspot target ranks the top approximately 5% of chromatic–brightness proxy pixels within each image. These are computed RGB image descriptors, not analytically calibrated hydrocarbon fractions. The post hoc map c/g check must not be described as prospective external validation. Equal-weight and exposure-corrected estimates target different image statistics.

## Publishing steps
1. Review and resolve permissions. Remove any file whose distribution is unauthorized.
2. Create or select a GitHub repository and upload the **extracted contents**, not this ZIP as a single opaque file.
3. Run dependency installation, scripts and software checks and record a tagged release. Supply restricted original navigation PNGs only to authorized evaluators.
4. Replace the manuscript's [GitHub URL / release identifier] placeholder with the actual public repository/release URL **after** upload is verified.
5. Ensure any promised data-access route in the manuscript is real, accurate and approved.
