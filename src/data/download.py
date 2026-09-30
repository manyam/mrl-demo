"""Downloader for UCSD Google Local Hawaii dataset.

Downloads meta-Hawaii.json.gz and review-Hawaii.json.gz with retry logic,
streaming disk writes, and integrity checks.
"""

import argparse
import gzip
import logging
import os
import sys
import time
from pathlib import Path
import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mrl.download")

DATASET_FILES = {
    "meta-Hawaii.json.gz": [
        "https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/meta-Hawaii.json.gz",
        "https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/meta-Hawaii.json.gz",
    ],
    "review-Hawaii.json.gz": [
        "https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/review-Hawaii.json.gz",
        "https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/review-Hawaii.json.gz",
    ],
}


def is_valid_gzip(filepath: Path) -> bool:
    """Check if file exists, is non-empty, and has a valid gzip header."""
    if not filepath.exists() or filepath.stat().st_size == 0:
        return False
    try:
        with gzip.open(filepath, "rb") as f:
            f.read(1024)
        return True
    except Exception as e:
        logger.warning(f"Integrity check failed for {filepath.name}: {e}")
        return False


def is_valid_json_file(filepath: Path) -> bool:
    """Check if an uncompressed json file exists and is non-empty."""
    return filepath.exists() and filepath.stat().st_size > 1024 * 1024


def download_file(urls: list[str], dest_path: Path, max_retries: int = 3, chunk_size: int = 65536) -> bool:
    """Download a file with retry logic and fallback mirrors."""
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    
    for url in urls:
        logger.info(f"Attempting download from: {url}")
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(url, stream=True, timeout=30)
                if response.status_code != 200:
                    logger.warning(f"HTTP {response.status_code} from {url}, attempt {attempt}/{max_retries}")
                    time.sleep(attempt * 2)
                    continue
                
                total_size = int(response.headers.get("content-length", 0))
                with open(temp_path, "wb") as f, tqdm(
                    desc=dest_path.name,
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
                
                # Check integrity if it's a gzip file
                if dest_path.name.endswith(".gz"):
                    if not is_valid_gzip(temp_path):
                        logger.error(f"Downloaded gzip file {dest_path.name} is corrupt.")
                        if temp_path.exists():
                            temp_path.unlink()
                        time.sleep(attempt * 2)
                        continue
                
                # Move completed file
                temp_path.replace(dest_path)
                logger.info(f"Successfully downloaded: {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.2f} MB)")
                return True
                
            except requests.RequestException as e:
                logger.warning(f"Request error downloading {url} (attempt {attempt}/{max_retries}): {e}")
                if temp_path.exists():
                    temp_path.unlink()
                time.sleep(attempt * 2)
    
    logger.error(f"Failed to download {dest_path.name} from all mirrors.")
    return False


def main():
    parser = argparse.ArgumentParser(description="Download UCSD Google Local Hawaii dataset.")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Path to raw data directory")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    success = True
    for filename, urls in DATASET_FILES.items():
        dest = raw_dir / filename
        
        # Check if already downloaded and valid
        if not args.force and is_valid_gzip(dest):
            logger.info(f"Skipping {filename}: already exists and passed integrity check ({dest.stat().st_size / (1024*1024):.2f} MB).")
            continue
            
        # Special check: if review-Hawaii.json.gz is needed but review-Hawaii_10.json exists (>100MB), treat as valid alternate
        if filename == "review-Hawaii.json.gz":
            alt_path = raw_dir / "review-Hawaii_10.json"
            if not args.force and is_valid_json_file(alt_path):
                logger.info(f"Found existing valid raw review file: {alt_path.name} ({alt_path.stat().st_size / (1024*1024):.2f} MB). Skipping {filename} download.")
                continue

        download_success = download_file(urls, dest)
        if not download_success:
            success = False

    if not success:
        logger.error("Data download pipeline encountered errors.")
        sys.exit(1)
        
    logger.info("All raw dataset files verified and ready.")


if __name__ == "__main__":
    main()
