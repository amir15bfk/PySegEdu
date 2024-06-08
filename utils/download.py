import os
import requests
import zipfile
import patoolib
from tqdm import tqdm

def download_file(url, output_path):
    """Download a file from a URL with a progress bar."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        with open(output_path, 'wb') as f, tqdm(
                desc=f"Downloading {os.path.basename(output_path)}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
                bar.update(len(chunk))
        print(f"Downloaded {output_path}")
    else:
        print(f"Failed to download the file from {url}. Status code: {response.status_code}")

def extract_zip(file_path, extract_to):
    """Extract a zip file."""
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {file_path} to {extract_to}")


def extract_rar(file_path, extract_to):
    patoolib.extract_archive(file_path, outdir=extract_to)


def download(path="./data"):
    # Create the data folder if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # URLs to download
    kvasir_seg_url = "https://datasets.simula.no/downloads/kvasir-seg.zip"
    pvt_model_url = "https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b3.pth"
    cvc_clinicdb_url = "https://www.dropbox.com/s/p5qe9eotetjnbmq/CVC-ClinicDB.rar?dl=1"

    # Paths to save the downloads
    kvasir_zip_path = os.path.join(path, "kvasir-seg.zip")
    pvt_model_path = os.path.join(path, "pvt_v2_b3.pth")
    cvc_rar_path = os.path.join(path, "CVC-ClinicDB.rar")

    # Download the kvasir-seg dataset
    download_file(kvasir_seg_url, kvasir_zip_path)

    # Unzip the kvasir-seg.zip file
    extract_zip(kvasir_zip_path, path)
    
    # Optionally, remove the zip file after extraction
    os.remove(kvasir_zip_path)
    print(f"Removed {kvasir_zip_path}")

    # Download the PVT model
    download_file(pvt_model_url, pvt_model_path)

    # Download the CVC-ClinicDB dataset
    download_file(cvc_clinicdb_url, cvc_rar_path)

    # Extract the CVC-ClinicDB.rar file
    extract_rar(cvc_rar_path, path)

    # Optionally, remove the rar file after extraction
    os.remove(cvc_rar_path)
    print(f"Removed {cvc_rar_path}")


if __name__=="__main__":
    download(path="./data")