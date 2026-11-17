# TESS eclipsing-binary regression data

Thirty unbinned mission cadences exercise ELISa's public `LCData` and CPU `Observer` APIs offline. Each star contributes five points around each eclipse. The selection uses target times and quality flags, so changes in model predictions cannot change the input sample. Each CSV keeps the archive cadence ID, time in BJD TDB minus 2,457,000, quality flag, flux and uncertainty in electrons/s.

| Target and study | TESS source | What the sample probes |
| --- | --- | --- |
| [V505 Per, Southworth](https://arxiv.org/abs/2106.04323) | TIC 348517784, Sector 18, 120 s SAP, DR73 | Narrow eclipses in a circular detached binary and uncertainties at two measured brightness levels. DR73 supersedes the photometry available for the 2021 study. |
| [AI Phe, Maxted et al.](https://arxiv.org/abs/2003.09295) | TIC 102069549, Sector 2, 120 s PDCSAP, DR42 | A strongly displaced secondary eclipse in an eccentric orbit. The current FITS timestamps and reduction differ from those used for the published Run A solution. |
| [NT Aps, Avramova-Boncheva et al.](https://doi.org/10.1017/pasa.2024.122) | TIC 402227266, Sector 66, 20 s PDCSAP, DR92 | Closely spaced eclipses of the observed contact binary. ELISa classifies the supplied 2023 median-input model as detached. A separate, explicitly circular sensitivity case uses common potential 2.44 and exercises the overcontact branch on the same timestamps. |

The assertions cover time-to-phase conversion, native relative flux and error handling, synthesis at measured times versus permuted or repeated phases, and the ordering of the observed and calculated flux on both sides of eclipse. The models use blackbody emission, fixed linear limb darkening and instantaneous samples. These are deliberately modest scientific assertions. Fits at the precision of the published radii or temperatures require matching the full light curve, LD law, exposure treatment, detrending and noise model.

One strict expected failure on the fork's `dev` branch records the magnitude-input uncertainty conversion. With equal magnitude errors, absolute flux uncertainties should scale with local brightness. The measured V505 eclipse-center/reference flux ratio is 0.619. This branch returns an error ratio of 1.000. Review the marker when the conversion is fixed. The native flux/error tests cover the quantities stored in these FITS files.

## Running and regenerating

From the repository root, with ELISa's test dependencies installed:

```sh
python -m pytest -q unittests/test_tess_binaries.py -rx
```

`products.json` pins the MAST identifiers, SHA-256 hashes, releases, sizes and cadence for the three excerpt inputs. It also includes the V505 Per target-pixel file for re-extracting SAP from calibrated pixels and the NT Aps Sector 65 light curve for comparison with the two-sector period study. The five full inputs total 86,785,920 bytes. To retrieve them and regenerate the checked-in tables into a separate directory:

```sh
python unittests/data/tess_binaries/fetch_products.py tess-fits
python unittests/data/tess_binaries/select_cadences.py --fits-dir tess-fits --output-dir regenerated
diff unittests/data/tess_binaries/v505_per_s18_lc.csv regenerated/v505_per_s18_lc.csv
diff unittests/data/tess_binaries/ai_phe_s02_lc.csv regenerated/ai_phe_s02_lc.csv
diff unittests/data/tess_binaries/nt_aps_s66_lc.csv regenerated/nt_aps_s66_lc.csv
```

Source observations are public NASA TESS/SPOC products obtained from MAST. Their calibration and pipeline processing precede these regression fixtures. [TESS data products](https://heasarc.gsfc.nasa.gov/docs/tess/data_products.html) · [MAST TESS release history](https://outerspace.stsci.edu/pages/viewpage.action?pageId=344591245)
