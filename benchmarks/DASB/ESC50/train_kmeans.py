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
import speechbrain.utils.parameter_transfer as sb_tf
from speechbrain.lobes.models.Cnn14 import Cnn14
from speechbrain.processing.features import STFT, Filterbank, spectral_magnitude

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
    feats = feats.permute(0, 2, 3, 1)
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
            batch = batch.to(device)
            X_stft = compute_stft(batch)
            X_stft_mag = spectral_magnitude(X_stft, power=0.5)
            net_input = compute_features(X_stft_mag)
            _, embeddings = embedding_model(net_input)

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


if __name__ == '__main__':

    # logger = logging.getLogger(__name__)
    import glob
    # get the dataloader
    paths = glob.glob('/data1/VGG-sound/vgg-sound-16k/**/*.ogg', recursive=True)

    from torch.utils.data import Dataset, DataLoader
    class VGGSound(Dataset):
        """Automatic Character Recognition Dataset"""

        def __init__(self, paths):
            self.paths = paths

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            wav, _ = torchaudio.load(paths[idx])
            return wav.squeeze()
    
    def collate_fn(wavs):
        """This function defines how we will """
        wavs_list = [wav for wav in wavs]

        nested_wavs = torch.nested.nested_tensor(wavs_list)
        wavs_padded = nested_wavs.to_padded_tensor(0)
        return wavs_padded

    tr_dataset = VGGSound(paths)
    tr_loader = DataLoader(tr_dataset, batch_size=32, collate_fn=collate_fn, shuffle=False)

    mel_bins = 80
    device = 'cuda'
    # load the model
    embedding_model = Cnn14(mel_bins=mel_bins, emb_dim=2048, return_reps=True).to(device)
    pretrainer = sb_tf.Pretrainer(collect_in='cnn14.ckpt', loadables={'embedding_model' : embedding_model},
                                                           paths={'embedding_model': 'speechbrain/cnn14-esc50/embedding_model.ckpt'})
    pretrainer.collect_files()
    pretrainer.load_collected()

    # define the feature extraction modules
    compute_stft = STFT(sample_rate=16000, win_length=23.2199, hop_length=11.6099, n_fft=1024)
    compute_features = Filterbank(n_mels=80, sample_rate=16000, n_fft=1024, log_mel=True)

    # define the kmeans parameters
    num_clusters = 100
    save_path = f'kmeans-cluster-{num_clusters}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for numlayer in range(4):
        print(f'Kmeans for layer {numlayer}')
        checkpoint_path = os.path.join(save_path, f"kmeans-cluster-{num_clusters}_layer_{numlayer}.pt")

        kmeans_model = fetch_kmeans_model(n_clusters=128, init='k-means++', max_iter=100, batch_size=1000, tol=0.0, max_no_improvement=100, n_init=20, reassignment_ratio=0.0, random_state=100, checkpoint_path=checkpoint_path)

        train(kmeans_model, tr_loader, embedding_model, save_path, numlayer, device=device, checkpoint_interval=100)

        print(f"Saving kmeans model at {checkpoint_path}.")
        save_model(kmeans_model, checkpoint_path)
