# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import zipfile
import requests
import itertools

LANGS = ["cy", "da", "es", "fr", "hi", "it", "ja", "ko", "mn", "pt", "sv", "sw", "ta", "th", "zh_CN", "zu"]

OUTDIR = "../datasets/MultiCCAligned"
BASEURL = "https://object.pouta.csc.fi/OPUS-MultiCCAligned/v1.1/moses"

os.makedirs(OUTDIR, exist_ok=True)

def download_and_unzip(url, zip_path, extract_dir):
    try:
        print(f"Downloading {url} ...")
        r = requests.get(url, stream=True, timeout=60)
        if r.status_code != 200:
            print(f"Failed: {url} (status {r.status_code})")
            return False

        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        print(f"Saved to {extract_dir}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

for src, tgt in itertools.combinations(LANGS, 2):
    pair = f"{src}-{tgt}"
    url = f"{BASEURL}/{pair}.txt.zip"
    zip_path = os.path.join(OUTDIR, f"{pair}.txt.zip")
    extract_dir = os.path.join(OUTDIR, pair)

    ok = download_and_unzip(url, zip_path, extract_dir)
    if not ok:
        print(f"Skipping {pair}")