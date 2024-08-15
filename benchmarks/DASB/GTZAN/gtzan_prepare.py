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
    # json_dict = {}

    # for ID, sample_metadata in metadata.items():
    #     fold_num = int(sample_metadata["fold"])
    #     if fold_num in folds_list:
    #         # Reading the signal (to retrieve duration in seconds)
    #         wav_file = os.path.join(
    #             os.path.abspath(audio_data_folder),
    #             # "fold" + str(fold_num) + "/",
    #             sample_metadata["filename"],
    #         )
    #         try:
    #             signal = read_audio(wav_file)
    #             file_info = torchaudio.info(wav_file)

    #             # If we're using sox/soundfile backend, file_info will have the old type
    #             if isinstance(file_info, torchaudio.AudioMetaData):
    #                 duration = signal.shape[0] / file_info.sample_rate
    #             else:
    #                 duration = signal.shape[0] / file_info[0].rate

    #             # Create entry for this sample ONLY if we have successfully read-in the file using SpeechBrain/torchaudio
    #             json_dict[ID] = {
    #                 "wav": sample_metadata["filename"],
    #                 "classID": int(sample_metadata["target"]),
    #                 "class_string": sample_metadata["class_string"],
    #                 # "salience": int(sample_metadata["salience"]),
    #                 "fold": sample_metadata["fold"],
    #                 "duration": duration,
    #             }
    #         except Exception:
    #             print(
    #                 f"There was a problem reading the file:{wav_file}. Skipping duration field for it."
    #             )
    #             logger.exception(
    #                 f"There was a problem reading the file:{wav_file}. Skipping it."
    #             )

    # # Writing the dictionary to the json file
    # # Need to make sure sub folder "manifest" exists, if not create it
    # parent_dir = os.path.dirname(json_file)
    # if not os.path.exists(parent_dir):
    #     os.mkdir(parent_dir)

    # with open(json_file, mode="w") as json_f:
    #     json.dump(json_dict, json_f, indent=2)

    



def check_folders(*folders):
    """Returns False if any passed folder does not exist.

    Arguments
    ---------
    *folders: list
        Folders to check.

    Returns
    -------
    pass: bool
    """
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True


def full_path_to_audio_file(data_folder, slice_file_name, fold_num):
    """Get path to file given slice file name and fold number

    Arguments
    ---------
    data_folder : str
        Folder that contains the dataset.
    slice_file_name : str
        Filename.
    fold_num : int
        Fold number.

    Returns
    -------
    string containing absolute path to corresponding file
    """
    return os.path.join(
        os.path.abspath(data_folder),
        "audio/",
        "fold" + str(fold_num) + "/",
        slice_file_name,
    )


def create_metadata_speechbrain_file(data_folder):
    """Get path to file given slice file name and fold number

    Arguments
    ---------
    data_folder : str
        ESC50 data folder.

    Returns
    -------
    string containing absolute path to metadata csv file modified for SpeechBrain or None if source file not found
    """
    import pandas as pd

    esc50_metadata_csv_path = os.path.join(
        os.path.abspath(data_folder), "meta/esc50.csv"
    )
    if not os.path.exists(esc50_metadata_csv_path):
        return None

    esc50_metadata_df = pd.read_csv(esc50_metadata_csv_path)
    # SpeechBrain wants an ID column
    esc50_metadata_df["ID"] = esc50_metadata_df.apply(
        lambda row: removesuffix(row["filename"], ".wav"), axis=1
    )
    esc50_metadata_df = esc50_metadata_df.rename(
        columns={"category": "class_string"}
    )

    esc50_speechbrain_metadata_csv_path = os.path.join(
        os.path.abspath(data_folder), "meta/", MODIFIED_METADATA_FILE_NAME
    )
    esc50_metadata_df.to_csv(esc50_speechbrain_metadata_csv_path, index=False)
    return esc50_speechbrain_metadata_csv_path


def removesuffix(some_string, suffix):
    """Removed a suffix from a string

    Arguments
    ---------
    some_string : str
        Any string.
    suffix : str
        Suffix to be removed from some_string.

    Returns
    -------
    string resulting from suffix removed from some_string, if found, unchanged otherwise
    """
    if some_string.endswith(suffix):
        return some_string[: -1 * len(suffix)]
    else:
        return some_string


def dataio_prep(hparams):
    "Creates the datasets and their data processing pipelines."

    data_audio_folder = hparams["audio_data_folder"]
    config_sample_rate = hparams["sample_rate"]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    hparams["resampler"] = torchaudio.transforms.Resample(
        new_freq=config_sample_rate
    )

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""

        wave_file = data_audio_folder + "/{:}".format(wav)

        sig, read_sr = torchaudio.load(wave_file)

        # If multi-channels, downmix it to a mono channel
        sig = torch.squeeze(sig)
        if len(sig.shape) > 1:
            sig = torch.mean(sig, dim=0)

        # Convert sample rate to required config_sample_rate
        if read_sr != config_sample_rate:
            # Re-initialize sampler if source file sample rate changed compared to last file
            if read_sr != hparams["resampler"].orig_freq:
                hparams["resampler"] = torchaudio.transforms.Resample(
                    orig_freq=read_sr, new_freq=config_sample_rate
                )
            # Resample audio
            sig = hparams["resampler"].forward(sig)

        sig = sig.float()
        sig = sig / sig.max()
        return sig

    # 3. Define label pipeline:
    @sb.utils.data_pipeline.takes("class_string")
    @sb.utils.data_pipeline.provides("class_string", "class_string_encoded")
    def label_pipeline(class_string):
        yield class_string
        class_string_encoded = label_encoder.encode_label_torch(class_string)
        yield class_string_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "class_string_encoded"],
        )

    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mappinng.
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="class_string",
    )

    return datasets, label_encoder
