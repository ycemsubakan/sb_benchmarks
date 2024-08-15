"""
Creates data manifest files for GTZAN
If the data does not exist in the specified --data_folder, we download the data automatically.

Authors:
 * Pooneh Mousavi2024
"""

import json
import logging
import os
import shutil
import requests
import tarfile

import torch
import torchaudio

import speechbrain as sb
from speechbrain.dataio.dataio import load_data_csv, read_audio
from speechbrain.utils.fetching import fetch
from speechbrain.utils.data_utils import get_all_files
logger = logging.getLogger(__name__)


import os

GTZAN_URL = "https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz"

def download_gtzan(data_path):
    """
    Downloads the GTZAN dataset from Hugging Face and extracts it to the specified directory.

    Parameters:
    - data_path (str): The directory where the extracted files will be saved.
    """
    if not os.path.exists(os.path.join(data_path, "genres")):
        logger.info(
            f"GTZAN is missing. We are now downloading it. Be patient it's a 1.2GB file."
        )
        # Send a GET request to download the file with streaming
        response = requests.get(GTZAN_URL, stream=True)

        # Check if the extraction directory exists; if not, create it
        if not os.path.exists(data_path):
            os.makedirs(data_path)
            logger.info(f"Directory '{data_path}' created.")

        # Open the tar.gz file and extract its contents to the specified directory
        with tarfile.open(fileobj=response.raw, mode="r|gz") as file:
            file.extractall(path=data_path)

        logger.info(f"Extraction complete. GTZAN are saved to '{data_path}'.")

        

def prepare_gtzan(
    data_folder,
    save_json_train,
    save_json_valid,
    save_json_test,
    skip_prep=False,
):
    """
    Prepares the json files for the ESC50 dataset.
    Prompts to download the dataset if it is not found in the `data_folder`.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the GTZAN dataset is stored.
    save_json_train : str
        Path where the train data specification file will be saved.
    save_json_valid : str
        Path where the validation data specification file will be saved.
    save_json_test : str
        Path where the test data specification file will be saved.
    skip_prep: bool
        If True, skip data preparation.

    Returns
    -------
    None

    Example
    -------
    >>> data_folder = '/path/to/genres'
    >>> def prepare_gtzan(data_folder, 'train.json', 'valid.json', 'test.json')
    """
    if skip_prep:
        return

    download_gtzan(data_folder)

    # If csv already exists, we skip the data preparation
    if skip(save_json_train, save_json_valid, save_json_test):

        msg = "%s already exists, skipping data preparation!" % (save_json_train)
        logger.info(msg)

        msg = "%s already exists, skipping data preparation!" % (save_json_valid)
        logger.info(msg)

        msg = "%s already exists, skipping data preparation!" % (save_json_test)
        logger.info(msg)

        return

    # List files and create manifest from list
    logger.info(
        f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}"
    )


    paths = get_all_files(os.path.join(data_folder,"genres"), match_and=['.wav'])
    # all_genres = list(set([path.split('/')[-2] for path in paths]))
    tr_pairs = [path for path in paths if int(path.split('.')[-2]) < 80]
    valid_pairs = [path for path in paths if 80 <= int(path.split('.')[-2]) < 90]
    test_pairs = [path for path in paths if 90 <= int(path.split('.')[-2]) < 100]
    # Creating json files
    create_json(save_json_train,tr_pairs)
    create_json(save_json_valid,valid_pairs)
    create_json(save_json_test,test_pairs)

def create_json(json_file, audiolist):
    """
    Creates the json file given a list of wav files.

    Arguments
    ---------
    metadata: dict
        A dictionary containing the ESC50 metadata file modified for the
        SpeechBrain, such that keys are IDs (which are the .wav file names without the file extension).
    audio_data_folder : str or Path
        Data folder that stores ESC50 samples.
    folds_list : list of int
        The list of folds [1,5] to include in this batch
    json_file : str
        The path of the output json file
    """
    # Processing all the wav files in the list

    json_dict = {}
    for audiofile in audiolist:
        try:
            # Getting info
            audioinfo = torchaudio.info(audiofile) # Your code here

            # Compute the duration in seconds.
            # This is the number of samples divided by the sampling frequency
            duration = audioinfo.num_frames / audioinfo.sample_rate # Your code here

            # Get genre Label by manipulating the audio path
            genre = audiofile.split('/')[-2] # Your code here (aim for 1 line)_

            # Get a unique utterance id
            uttid = audiofile.split('/')[-1] # Your code here (aim for 1 line)

            # Create entry for this utterance
            json_dict[uttid] = {
                "wav": audiofile,
                "length": duration,
                "genre": genre,
            }
        except Exception:
            print(
                f"There was a problem reading the file:{audiofile}. Skipping duration field for it."
            )
            logger.exception(
                f"There was a problem reading the file:{audiofile}. Skipping it."
            )
    
    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
      json.dump(json_dict, json_f, indent=2)
    logger.info(f"{json_file} successfully created!")

def skip(save_csv_train, save_csv_dev, save_csv_test):
    """
    Detects if the Common Voice data preparation has been already done.
    If the preparation has been done, we can skip it.
    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    # Checking folders and save options
    skip = False

    if (
        os.path.isfile(save_csv_train)
        and os.path.isfile(save_csv_dev)
        and os.path.isfile(save_csv_test)
    ):
        skip = True

    return skip