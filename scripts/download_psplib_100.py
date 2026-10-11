"""
Download exactly 25 diverse genuine PSPLIB instances for each family:
j30, j60, j90, j120 (total 100 unique instances).

Spans the parameter grid (NC, RF, RS) uniformly.
"""

import io
import os
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://www.om-db.wi.tum.de/psplib/"
DEST_DIR = Path("data/instances")
DEST_DIR.mkdir(parents=True, exist_ok=True)

# 25 parameter settings for 48-setting sets (j30, j60, j90)
PARAMS_48 = [
    1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29,
    31, 33, 35, 37, 39, 41, 43, 45, 47, 48
]

# 25 parameter settings for 60-setting sets (j120)
# (60 settings = 3 NC * 4 RF * 5 RS)
PARAMS_60 = [
    1, 3, 6, 8, 11, 13, 16, 18, 21, 23, 26, 28, 31, 33, 36,
    38, 41, 43, 46, 48, 51, 53, 56, 58, 60
]

SETS_CONFIG = [
    ("j30", PARAMS_48),
    ("j60", PARAMS_48),
    ("j90", PARAMS_48),
    ("j120", PARAMS_60),
]


def download_and_extract() -> None:
    total_saved = 0
    for set_name, param_indices in SETS_CONFIG:
        url = f"{BASE_URL}download_dataset.php?set={set_name}&mode=sm&format=zip"
        print(f"Downloading {set_name}.zip from {url}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = resp.read()
        print(f"  Downloaded {len(data)} bytes. Extracting 25 selected instances...")

        with zipfile.ZipFile(io.BytesIO(data)) as z:
            namelist = z.namelist()
            saved_for_set = 0
            for p_idx in param_indices:
                target_filename = f"{set_name}{p_idx}_1.sm"
                # Search case-insensitively or exact
                matched = [n for n in namelist if Path(n).name.lower() == target_filename.lower()]
                if not matched:
                    # fallback to any instance in that parameter group
                    prefix = f"{set_name}{p_idx}_"
                    matched = [n for n in namelist if Path(n).name.lower().startswith(prefix.lower())]

                if matched:
                    chosen = matched[0]
                    dest_file = DEST_DIR / Path(chosen).name
                    with open(dest_file, "wb") as f_out:
                        f_out.write(z.read(chosen))
                    saved_for_set += 1
                else:
                    print(f"  Warning: No match found for {target_filename} in {set_name}")

            print(f"  Saved {saved_for_set} instances for {set_name}.")
            total_saved += saved_for_set

    print(f"\nDone! Total instances saved in {DEST_DIR}: {total_saved}")


if __name__ == "__main__":
    download_and_extract()
