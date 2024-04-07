import os
import tarfile

import requests


def download_and_extract(url, extract_to='data'):
    """
    Download a tar.gz file from a given URL and extract it to a specified directory.

    Args:
    - url (str): URL of the tar.gz file to download.
    - extract_to (str): Directory path where the contents will be extracted.
    """
    # Make sure the directory exists
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    # Download the file
    print("Downloading dataset..")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        # Define the local filename to save data
        file_path = os.path.join(extract_to, url.split('/')[-1])

        # Write the downloaded content to a file
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    file.write(chunk)

        # Extract the tar.gz file
        print("Extracting file..")
        if file_path.endswith('.tar.gz'):
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(path=extract_to)

        print(f"File downloaded and extracted to {extract_to}")
    else:
        print("Failed to download the file")


# URL of the file you want to download and extract
url = 'http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz'

# Call the function
download_and_extract(url)
