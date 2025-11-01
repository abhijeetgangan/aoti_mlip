import logging
import os
import shutil
import urllib.request

logger = logging.getLogger(__name__)


def download_file(url: str, output_path: str, timeout: int = 30):
    """
    A wrapper around requests.get to download a file from a URL.

    Args:
        url (str): The URL to download the file from.
        output_path (str): The path to save the downloaded file to.
        timeout (int): The timeout for the download in seconds.
    """

    logger.info(f"Downloading file from {url} to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(output_path, "wb") as f:
            shutil.copyfileobj(resp, f)  # streams without loading whole file into memory
            logger.info(f"File downloaded to {output_path}")
    except Exception as e:
        logger.error(f"Error downloading file from {url}: {e}")
        raise e


def download_mattersim_checkpoint(
    checkpoint_name: str, save_folder: str = "~/.local/mattersim/pretrained_models/"
):
    """
    Download a checkpoint from the Microsoft Mattersim repository.

    Args:
        checkpoint_name (str): The name of the checkpoint to download.
        save_folder (str): The local folder to save the checkpoint to.
    """

    GITHUB_CHECKPOINT_PREFIX = (
        "https://raw.githubusercontent.com/microsoft/mattersim/main/pretrained_models/"
    )
    checkpoint_url = GITHUB_CHECKPOINT_PREFIX + checkpoint_name.strip("/")
    save_path = os.path.join(os.path.expanduser(save_folder), checkpoint_name.strip("/"))
    download_file(checkpoint_url, save_path)
