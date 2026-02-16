import os
import requests
import zipfile
from tqdm import tqdm

def download_file(url, dest_path):
    """
    Downloads a file from a URL to a specific destination with a progress bar.
    
    Args:
        url (str): The URL of the file to download.
        dest_path (str): The local path where the file should be saved.
    """
    if os.path.exists(dest_path):
        print(f"{dest_path} already exists. Skipping download.")
        return

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    print(f"Downloading {os.path.basename(dest_path)}...")
    with open(dest_path, 'wb') as file, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def extract_zip(zip_path, extract_to):
    """
    Extracts a zip file to a specified directory.
    
    Args:
        zip_path (str): Path to the zip file.
        extract_to (str): Directory where contents should be extracted.
    """
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")

def main():
    """
    Main execution flow for setting up the COCO validation dataset.
    Establishes the 'data' directory structure and fetches necessary files.
    """
    # URLs for COCO 2017 Validation set and Annotations
    # Source: http://cocodataset.org/#download
    val_images_url = "http://images.cocodataset.org/zips/val2017.zip"
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

    # Define directories
    base_dir = "data"
    raw_dir = os.path.join(base_dir, "raw")
    
    os.makedirs(raw_dir, exist_ok=True)

    # Define file paths
    val_zip_path = os.path.join(raw_dir, "val2017.zip")
    ann_zip_path = os.path.join(raw_dir, "annotations.zip")

    # 1. Download Files
    download_file(val_images_url, val_zip_path)
    download_file(annotations_url, ann_zip_path)

    # 2. Extract Files
    extract_zip(val_zip_path, raw_dir)
    extract_zip(ann_zip_path, raw_dir)

    print("\nDataset setup complete!")
    print(f"Images located at: {os.path.join(raw_dir, 'val2017')}")
    print(f"Annotations located at: {os.path.join(raw_dir, 'annotations')}")

if __name__ == "__main__":
    main()
