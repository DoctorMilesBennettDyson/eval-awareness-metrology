"""Download external inputs into data/external/ (not redistributed by this repository).

- Contrastive set of Nguyen et al. 2025 (github.com/Jordine/pivotal-test-phase-steering, no license file)
- SAD stages zip (github.com/LRudL/sad). SAD texts must stay encrypted; this script only stores the zip.
- The Devbunova 2x2 dataset (MIT) is fetched by data.py from the Hugging Face Hub.
"""
import urllib.request

from data import EXT

FILES = {
    "jordine_contrastive_dataset.json":
        "https://raw.githubusercontent.com/Jordine/pivotal-test-phase-steering/main/datasets/contrastive_dataset.json",
    "sad_stages_private_data_gen.zip":
        "https://github.com/LRudL/sad/raw/main/sad/stages/private_data_gen.zip",
}

if __name__ == "__main__":
    EXT.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dst = EXT / name
        if not dst.exists():
            urllib.request.urlretrieve(url, dst)
        print(name, dst.stat().st_size, "bytes")
