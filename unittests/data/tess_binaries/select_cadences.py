"""Regenerate the small TESS excerpts from the pinned, locally downloaded FITS files."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

HERE = Path(__file__).resolve().parent
# Each pair is (primary, secondary) in BTJD/TDB, followed by sample offsets in days.
# V505 uses Table II's circular ephemeris. AI Phe uses the Run A eccentric orbit
# to locate the secondary near BTJD 1374.089. NT Aps uses two Table 3 minima.
TARGETS = {
    "v505_per_s18_lc": ((1798.516720, 1800.6277308), (-0.15, -0.05, 0, 0.05, 0.15)),
    "ai_phe_s02_lc": ((1362.82850, 1374.089), (-0.55, -0.18, 0, 0.18, 0.55)),
    "nt_aps_s66_lc": ((3120.67442, 3120.82168), (-0.066, -0.028, 0, 0.028, 0.066)),
}
POSITIONS = ("left_reference", "left_shoulder", "center", "right_shoulder", "right_reference")
FIELDS = ("eclipse", "position", "cadence", "time_btjd", "quality", "flux_e_s", "error_e_s")


def extract(product, path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != product["sha256"]:
        raise ValueError(f"Wrong input bytes for {path.name}")
    with fits.open(path) as hdus:
        header, lc = hdus[0].header, hdus["LIGHTCURVE"]
        if (header["TICID"], header["SECTOR"], header["DATA_REL"]) != (
            product["tic"], product["sector"], product["data_release"]
        ):
            raise ValueError(f"Unexpected target, sector or release in {path.name}")
        if lc.header["TIMESYS"] != "TDB" or lc.header["BJDREFI"] + lc.header["BJDREFF"] != 2457000:
            raise ValueError(f"Expected BTJD/TDB timestamps in {path.name}")
        col = product["flux_column"]
        data = lc.data
        finite = np.isfinite(data["TIME"]) & np.isfinite(data[col]) & np.isfinite(data[col + "_ERR"])
        good = data[finite & (data["QUALITY"] == 0) & (data[col] > 0) & (data[col + "_ERR"] > 0)]
        centers, offsets = TARGETS[product["id"]]
        rows = []
        for eclipse, center in zip(("primary", "secondary"), centers):
            for position, offset in zip(POSITIONS, offsets):
                target = center + offset
                row = good[np.argmin(abs(good["TIME"] - target))]
                if abs(row["TIME"] - target) * 86400 > 0.5 * product["cadence_seconds"] + 1.0:
                    raise ValueError(f"Missing cadence at {target} in {path.name}")
                rows.append((
                    eclipse, position, int(row["CADENCENO"]), repr(float(row["TIME"])), int(row["QUALITY"]),
                    repr(float(row[col])), repr(float(row[col + "_ERR"])),
                ))
        if len({row[2] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate selected cadence in {path.name}")
        return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--manifest", type=Path, default=HERE / "products.json")
    args = parser.parse_args()
    products = {row["id"]: row for row in json.loads(args.manifest.read_text())["products"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for product_id in TARGETS:
        product = products[product_id]
        rows = extract(product, args.fits_dir / product["filename"])
        with (args.output_dir / f"{product_id}.csv").open("w", newline="") as output:
            writer = csv.writer(output, lineterminator="\n")
            writer.writerow(FIELDS)
            writer.writerows(rows)


if __name__ == "__main__":
    main()
