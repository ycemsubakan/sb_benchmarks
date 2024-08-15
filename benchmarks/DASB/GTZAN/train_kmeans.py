"""
Utilities for training kmeans model.

Author
 * Pooneh Mousavi 2023
"""

import logging
import os

from tqdm.contrib import tqdm
import torchaudio
import torch
import speechbrain as sb
from speechbrain.lobes.models.Cnn14 import Cnn14
from speechbrain.processing.features import STFT, Filterbank, spectral_magnitude
from speechbrain.dataio.dataloader import make_dataloader

try:
    from sklearn.cluster import MiniBatchKMeans
except ImportError:
    err_msg = "The optional dependency sklearn is needed to use this module\n"
    err_msg += "Cannot import sklearn.cluster.MiniBatchKMeans to use KMeans/\n"
    err_msg += "Please follow the instructions below\n"
    err_msg += "=============================\n"
    err_msg += "pip install -U scikit-learn\n"
    raise ImportError(err_msg)
import joblib



def accumulate_and_extract_features(
    batch, features_list, ssl_model, ssl_layer_num, device
):
    """Extract features (output of SSL model) and acculamte them on cpu to be used for clustering.

    Arguments
    ---------
        batch: tensor
            Single batch of data.
        features_list : list
            accumulate features list.
        ssl_model
            SSL-model used to  extract features used for clustering.
        ssl_layer_num: int
            specify output of which layer of the ssl_model should be used.
        device
            CPU or  GPU.
    """
    batch = batch.to(device)
    wavs, wav_lens = batch.sig
    wavs, wav_lens = (
        wavs.to(device),
        wav_lens.to(device),
    )
    feats = ssl_model(wavs, wav_lens)[ssl_layer_num].flatten(end_dim=-2)
    features_list.extend(feats.to("cpu").detach().numpy())


def accumulate_and_extract_features_cnn14(feats, features_list):
    # flatten over time and frequency
    feats = feats.reshape(-1, feats.shape[-1])
    features_list.extend(feats.to("cpu").detach().numpy())


def fetch_kmeans_model(
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,
    random_state,
    checkpoint_path,
):
    """Return a k-means clustering model with specified parameters.

    Arguments
    ---------
        n_clusters : MiniBatchKMeans
            The number of clusters to form as well as the number of centroids to generate.
        init : int
            Method for initialization: {'k-means++'', ''random''}
        max_iter : int
            Maximum number of iterations over the complete dataset before stopping independently of any early stopping criterion heuristics.
        batch_size : int
            Size of the mini batches.
        tol : float
            Control early stopping based on the relative center changes as measured by a smoothed, variance-normalized of the mean center squared position changes.
        max_no_improvement :int
            Control early stopping based on the consecutive number of mini batches that does not yield an improvement on the smoothed inertia.
        n_init : int
            Number of random initializations that are tried
        reassignment_ratio : float
            Control the fraction of the maximum number of counts for a center to be reassigned.
        random_state :int
            Determines random number generation for centroid initialization and random reassignment.
        compute_labels : bool
            Compute label assignment and inertia for the complete dataset once the minibatch optimization has converged in fit.
        init_size : int
            Number of samples to randomly sample for speeding up the initialization.
        checkpoint_path : str
            Path to saved model.

    Returns
    ---------
        MiniBatchKMeans
            a k-means clustering model with specified parameters.
    """
    if os.path.exists(checkpoint_path):
        print(f"The checkpoint is loaded from {checkpoint_path}.")
        return joblib.load(checkpoint_path)

    print(
        f"No checkpoint is found at {checkpoint_path}. New model is initialized for training."
    )
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        tol=tol,
        max_no_improvement=max_no_improvement,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
        random_state=random_state,
        verbose=1,
        compute_labels=True,
        init_size=None,
    )


def process_chunks(data, chunk_size, model):
    """Process data in chunks of a specified size.

    Args:
        data (list): 
            The list of integers to be processed.
        chunk_size (int): 
            The size of each chunk.
        model : MiniBatchKMeans
            The initial kmeans model for training.

    return:
        model : MiniBatchKMeans
            The initial kmeans model for training.
    """
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]

        # Skip processing if the chunk size is smaller than chunk_size
        if len(chunk) < chunk_size:
            break
        model = model.partial_fit(chunk)


def train(
    model,
    train_set,
    ssl_model,
    save_path,
    ssl_layer_num,
    kmeans_batch_size=1000,
    device="cpu",
    checkpoint_interval=10,
):
    """Train a  Kmeans model .

    Arguments
    ---------
        model : MiniBatchKMeans
            The initial kmeans model for training.
        train_set : Dataloader
            Batches of tarining data.
        ssl_model
            SSL-model used to  extract features used for clustering.
        checkpoint_path: string
            Path to save intra-checkpoints and dataloader.
        ssl_layer_num : int
            Specify output of which layer of the ssl_model should be used.
        device
            CPU or  GPU.
        kmeans_batch_size : int
            Size of the mini batches.
        checkpoint_interval: int
            Determine at which iterations to save the checkpoints.
    """
    print("Start training kmeans model.")
    features_list = []
    iteration = 0

    with tqdm(
        train_set,
        dynamic_ncols=True,
    ) as t:
        for batch in t:
            wavs = batch.sig.data.to(device)
            embeddings = embedding_model(wavs)

            # extract features from the SSL model
            accumulate_and_extract_features_cnn14(embeddings[ssl_layer_num], features_list)

            # train a kmeans model on a single batch if features_list reaches the kmeans_batch_size.
            if len(features_list) >= kmeans_batch_size:
                process_chunks(features_list, kmeans_batch_size, model)
                iteration += 1
                features_list = []

            if (iteration + 1) % checkpoint_interval == 0:
                print(
                    f"Saving intra-checkpoints for iteration {iteration}."
                )
                # train_set._speechbrain_save(
                #     os.path.join(save_path, "dataloader-TRAIN.ckpt")
                # )
                # checkpoint_path = os.path.join(
                #    save_path,
                #    f"kmeans-cluster-{model.n_clusters}-layer-{ssl_layer_num}.pt",
                # )
                save_model(model, checkpoint_path)

        if len(features_list) >= kmeans_batch_size:
            process_chunks(features_list, kmeans_batch_size, model)


def save_model(model, checkpoint_path):
    """Save a  Kmeans model .

    Arguments
    ---------
        model : MiniBatchKMeans
            The  kmeans model to be saved.
        checkpoint_path : str)
            Path to save the model..
    """
    joblib.dump(model, open(checkpoint_path, "wb"))


def dataio_prep(sample_rate=24000, data_folder='/data2/GTZAN_v2'):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined
    functions. We expect `prepare_mini_librispeech` to have been called before
    this, so that the `train.json`, `valid.json`,  and `valid.json` manifest
    files are available.
    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.
    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "valid" that correspond
        to the appropriate DynamicItemDataset object.
    """

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        sig = sb.dataio.dataio.read_audio(wav)
        info = torchaudio.info(wav)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate, sample_rate,
        )(sig)
        return resampled

    # Initialization of the label encoder. The label encoder assignes to each
    # of the observed label a unique index (e.g, 'blues': 0, 'rock': 1, ..)
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("genre")
    @sb.utils.data_pipeline.provides("genre", "genre_encoded")
    def label_pipeline(genre):
        yield genre
        genre_encoded = label_encoder.encode_label_torch(genre)
        yield genre_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    data_info = {
        "train": "train.json",
        "valid": "valid.json",
        "test": "test.json",
    }
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": data_folder},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "genre_encoded"],
        )
    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mappinng.

    lab_enc_file = os.path.join('.', "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="genre",
    )
    return datasets

if __name__ == '__main__':

    # logger = logging.getLogger(__name__)
    # get the dataloader
   

    from gtzan_prepare import prepare_gtzan  # noqa E402

    # Data preparation, to be run on only one process.
    sb.utils.distributed.run_on_main(
        prepare_gtzan,
        kwargs={
            "data_folder": '/data2/GTZAN_v2',
            "save_json_train": 'train.json',
            "save_json_valid": 'valid.json',
            "save_json_test": 'test.json',
        },
    )

    # Data preparation, to be run on only one process.
    # Create dataset objects "train", "valid", and "test".
    datasets = dataio_prep()
    tr_loader = make_dataloader(datasets['train'], batch_size=16)


    device = 'cuda'
    from MERT import MERT
    # load the model
    embedding_model = MERT(source='m-a-p/MERT-v1-330M', save_path='MERT-files', freeze=True, output_all_hiddens=True).to(device)
    
    # define the kmeans parameters
    num_layers = [0, 2, 6, 11, 17, 22]
    num_clusters = 128
    save_path = f'kmeans-cluster-{num_clusters}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for numlayer in num_layers:
        print(f'Kmeans for layer {numlayer}')
        checkpoint_path = os.path.join(save_path, f"kmeans-cluster-{num_clusters}_layer_{numlayer}.pt")

        kmeans_model = fetch_kmeans_model(n_clusters=128, init='k-means++', max_iter=100, batch_size=1000, tol=0.0, max_no_improvement=100, n_init=20, reassignment_ratio=0.0, random_state=100, checkpoint_path=checkpoint_path)

        train(kmeans_model, tr_loader, embedding_model, save_path, numlayer, device=device, checkpoint_interval=100)

        print(f"Saving kmeans model at {checkpoint_path}.")
        save_model(kmeans_model, checkpoint_path)
