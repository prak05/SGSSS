"""
Script to download real-world benchmark datasets from the net:
1. Patterson RCPSP instances (RG30 suite, Pat1.rcp to Pat30.rcp)
2. Additional PSPLIB instances (j30, j60, j90, j120)
All files are downloaded from verified open-source operational research repositories.
"""

import os
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = PROJECT_ROOT / "data" / "instances" / "net_resources"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

# 1. Patterson benchmark suite from RG30 (sydney-machine-learning/MEHH_RCPSP)
PATTERSON_BASE_URL = "https://raw.githubusercontent.com/sydney-machine-learning/MEHH_RCPSP/master/datasets/RG30/set1"
PATTERSON_FILES = [f"Pat{i}.rcp" for i in range(1, 31)]

# 2. PSPLIB benchmark instances from sydney-machine-learning/MEHH_RCPSP
PSPLIB_SUITES = {
    "j30": [f"j301_{i}.sm" for i in range(1, 11)] + [f"j302_{i}.sm" for i in range(1, 11)],
    "j60": [f"j601_{i}.sm" for i in range(1, 11)] + [f"j602_{i}.sm" for i in range(1, 11)],
    "j90": [f"j901_{i}.sm" for i in range(1, 11)] + [f"j902_{i}.sm" for i in range(1, 11)],
    "j120": [f"j1201_{i}.sm" for i in range(1, 11)] + [f"j1202_{i}.sm" for i in range(1, 11)],
}

def download_file(url: str, dest_path: Path) -> bool:
    try:
        urllib.request.urlretrieve(url, dest_path)
        return True
    except Exception as e:
        print(f"[-] Failed to download {url}: {e}")
        return False

def main():
    print(f"[*] Downloading new benchmark resources from the net into: {TARGET_DIR}")
    downloaded_count = 0

    # Download Patterson instances
    print(f"\n--- 1. Downloading Patterson Benchmark Instances (.rcp) ---")
    for fname in PATTERSON_FILES:
        dest = TARGET_DIR / fname
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[+] Already exists: {fname}")
            downloaded_count += 1
            continue
        url = f"{PATTERSON_BASE_URL}/{fname}"
        if download_file(url, dest):
            print(f"[+] Downloaded: {fname} ({dest.stat().st_size} bytes)")
            downloaded_count += 1

    # Download PSPLIB instances
    print(f"\n--- 2. Downloading PSPLIB Benchmark Instances (.sm) ---")
    for suite, files in PSPLIB_SUITES.items():
        base_url = f"https://raw.githubusercontent.com/sydney-machine-learning/MEHH_RCPSP/master/datasets/{suite}"
        for fname in files:
            dest = TARGET_DIR / fname
            if dest.exists() and dest.stat().st_size > 0:
                print(f"[+] Already exists: {fname}")
                downloaded_count += 1
                continue
            url = f"{base_url}/{fname}"
            if download_file(url, dest):
                print(f"[+] Downloaded: {fname} ({dest.stat().st_size} bytes)")
                downloaded_count += 1

    print(f"\n[✓] Finished! Total net resources ready for training: {downloaded_count} files in {TARGET_DIR}")

if __name__ == "__main__":
    main()
