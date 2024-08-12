#!/usr/bin/python3

"""Recipe to train a classifier on ESC50 data.

To run this recipe, use the following command:
> python train.py hparams/<config>.yaml --data_folder yourpath/ESC-50-master

Authors
    * Cem Subakan 2022, 2023
    * Francesco Paissan 2022, 2023
    * Luca Della Libera 2024

Based on the Urban8k recipe by
    * David Whipps 2021
    * Ala Eddine Limame 2021
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchvision
from confusion_matrix_fig import create_cm_fig
from esc50_prepare import prepare_esc50
from hyperpyyaml import load_hyperpyyaml
from sklearn.metrics import confusion_matrix
from wham_prepare import combine_batches, prepare_wham

import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.tokenizers.discrete_SSL_tokenizer import DiscreteSSLTokenizer
import glob
import joblib


class ESC50Brain(sb.core.Brain):
    """Class for classifier training."""

    def compute_forward(self, batch, stage):
        """Computation pipeline based on an encoder + sound classifier."""
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        # Augment if specified
        if hasattr(self.hparams, "augmentation") and stage == sb.Stage.TRAIN:
            wavs, lens = self.hparams.augmentation(wavs, lens)

        # augment batch with WHAM!
        if hasattr(self.hparams, "add_wham_noise"):
            if self.hparams.add_wham_noise:
                wavs = combine_batches(wavs, iter(self.hparams.wham_dataset))

        X_stft = self.modules.compute_stft(wavs)
        net_input = sb.processing.features.spectral_magnitude(
            X_stft, power=self.hparams.spec_mag_power
        )
        if (
            hasattr(self.hparams, "use_melspectra")
            and self.hparams.use_melspectra
        ):
            net_input = self.modules.compute_fbank(net_input)

        if (not self.hparams.use_melspectra) or self.hparams.use_log1p_mel:
            net_input = torch.log1p(net_input)

        # Embeddings + sound classifier
        if hasattr(self.modules.embedding_model, "config"):
            # Hugging Face model
            config = self.modules.embedding_model.config
            # Resize to match expected resolution
            net_input = torchvision.transforms.functional.resize(
                net_input, (config.image_size, config.image_size)
            )
            # Expand to have 3 channels
            net_input = net_input[:, None, ...].expand(-1, 3, -1, -1)
            if config.model_type == "focalnet":
                embeddings = self.modules.embedding_model(
                    net_input
                ).feature_maps[-1]
                embeddings = embeddings.mean(dim=(-1, -2))
            elif config.model_type == "vit":
                embeddings = self.modules.embedding_model(
                    net_input
                ).last_hidden_state.movedim(-1, -2)
                embeddings = embeddings.mean(dim=-1)
            else:
                raise NotImplementedError
        else:
            # SpeechBrain model
            embeddings, h = self.modules.embedding_model(net_input)

        with torch.no_grad():
            self.hparams.codec.to(self.device).eval()
            tokens, _ = self.hparams.codec(h)

        embeddings = self.modules.discrete_embedding_layer(tokens)
        att_w = self.modules.attention_mlp(embeddings)
        feats = torch.matmul(att_w.transpose(2, -1), embeddings).squeeze(-2)
        # last dim will be used for AdaptativeAVG pool
        outputs = self.hparams.avg_pool(feats, lens)
        outputs = outputs.view(outputs.shape[0], -1)
        outputs = self.modules.classifier(outputs).unsqueeze(1)
        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using class-id as label."""
        predictions, lens = predictions
        uttid = batch.id
        classid, _ = batch.class_string_encoded

        # Target augmentation
        N_augments = int(predictions.shape[0] / classid.shape[0])
        classid = torch.cat(N_augments * [classid], dim=0)

        # loss = self.hparams.compute_cost(predictions.squeeze(1), classid, lens)
        target = F.one_hot(
            classid.squeeze(), num_classes=self.hparams.out_n_neurons
        )
        loss = (
            -(F.log_softmax(predictions.squeeze(1), 1) * target).sum(1).mean()
        )

        if stage != sb.Stage.TEST:
            if hasattr(self.hparams.lr_annealing, "on_batch_end"):
                self.hparams.lr_annealing.on_batch_end(self.optimizer)

        # Append this batch of losses to the loss metric
        self.loss_metric.append(
            uttid, predictions, classid, lens, reduction="batch"
        )

        # Confusion matrices
        if stage != sb.Stage.TRAIN:
            y_true = classid.cpu().detach().numpy().squeeze(-1)
            y_pred = predictions.cpu().detach().numpy().argmax(-1).squeeze(-1)

        if stage == sb.Stage.VALID:
            confusion_matix = confusion_matrix(
                y_true,
                y_pred,
                labels=sorted(self.hparams.label_encoder.ind2lab.keys()),
            )
            self.valid_confusion_matrix += confusion_matix
        if stage == sb.Stage.TEST:
            confusion_matix = confusion_matrix(
                y_true,
                y_pred,
                labels=sorted(self.hparams.label_encoder.ind2lab.keys()),
            )
            self.test_confusion_matrix += confusion_matix

        # Compute accuracy using MetricStats
        self.acc_metric.append(
            uttid, predict=predictions, target=classid, lengths=lens
        )

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions, classid, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )

        # Compute accuracy using MetricStats
        # Define function taking (prediction, target, length) for eval
        def accuracy_value(predict, target, lengths):
            """Computes accuracy."""
            nbr_correct, nbr_total = sb.utils.Accuracy.Accuracy(
                predict, target, lengths
            )
            acc = torch.tensor([nbr_correct / nbr_total])
            return acc

        self.acc_metric = sb.utils.metric_stats.MetricStats(
            metric=accuracy_value, n_jobs=1
        )

        # Confusion matrices
        if stage == sb.Stage.VALID:
            self.valid_confusion_matrix = np.zeros(
                shape=(self.hparams.out_n_neurons, self.hparams.out_n_neurons),
                dtype=int,
            )
        if stage == sb.Stage.TEST:
            self.test_confusion_matrix = np.zeros(
                shape=(self.hparams.out_n_neurons, self.hparams.out_n_neurons),
                dtype=int,
            )

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        # Compute/store important stats
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            self.train_stats = {
                "loss": self.train_loss,
                "acc": self.acc_metric.summarize("average"),
            }
        # Summarize Valid statistics from the stage for record-keeping
        elif stage == sb.Stage.VALID:
            valid_stats = {
                "loss": stage_loss,
                "acc": self.acc_metric.summarize("average"),
                "error": self.error_metrics.summarize("average"),
            }
        # Summarize Test statistics from the stage for record-keeping
        else:
            test_stats = {
                "loss": stage_loss,
                "acc": self.acc_metric.summarize("average"),
                "error": self.error_metrics.summarize("average"),
            }

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # Tensorboard logging
            if self.hparams.use_tensorboard:
                self.hparams.tensorboard_train_logger.log_stats(
                    stats_meta={"Epoch": epoch},
                    train_stats=self.train_stats,
                    valid_stats=valid_stats,
                )
                # Log confusion matrix fig to tensorboard
                cm_fig = create_cm_fig(
                    self.valid_confusion_matrix,
                    display_labels=list(
                        self.hparams.label_encoder.ind2lab.values()
                    ),
                )
                self.hparams.tensorboard_train_logger.writer.add_figure(
                    "Validation Confusion Matrix", cm_fig, epoch
                )

            # The train_logger writes a summary to stdout and to the log file
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=valid_stats,
            )
            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(
                meta=valid_stats, min_keys=["error"]
            )

        # We also write statistics about test data to stdout and to the log file
        if stage == sb.Stage.TEST:
            # Per class accuracy from Test confusion matrix
            per_class_acc_arr = np.diag(self.test_confusion_matrix) / np.sum(
                self.test_confusion_matrix, axis=1
            )
            per_class_acc_arr_str = "\n" + "\n".join(
                "{:}: {:.3f}".format(class_id, class_acc)
                for class_id, class_acc in enumerate(per_class_acc_arr)
            )

            self.hparams.train_logger.log_stats(
                {
                    "Epoch loaded": self.hparams.epoch_counter.current,
                    "\n Per Class Accuracy": per_class_acc_arr_str,
                    "\n Confusion Matrix": "\n{:}\n".format(
                        self.test_confusion_matrix
                    ),
                },
                test_stats=test_stats,
            )


def dataio_prep(hparams):
    """Creates the datasets and their data processing pipelines."""
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
        """The label pipeline."""
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
    # mapping.
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="class_string",
    )

    return datasets, label_encoder


class DiscreteSSL_ESC50(nn.Module):
    """This lobe enables the integration of HuggingFace and SpeechBrain
    pretrained Discrete SSL models.

    Transformer from HuggingFace needs to be installed:
    https://huggingface.co/transformers/installation.html

    The model can be used as a fixed Discrete feature extractor or can be finetuned. It
    will download automatically the model from HuggingFace or use a local path.

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "facebook/hubert-base-ls960"
    save_path : str
        Path (dir) of the downloaded model.
    ssl_model : str
        SSL model to extract semantic tokens from its layers' output. Note that output_all_hiddens should be set to True to enable multi-layer discretenation.
    kmeans_repo_id : str
        Huggingface repository that contains the pre-trained k-means models.
    kmeans_dataset : str
        Name of the dataset that Kmeans model on HF repo is trained with.
    num_clusters:  int or List[int] (default: 1000)
            determine the number of clusters of the targeted kmeans models to be downloaded. It could be varying for each layer.
    layers_num: : List[int] (Optional)
            detremine layers to be download from HF repo. If it is not provided, all layers with num_clusters(int) is loaded from HF repo. If num_clusters is a list, the layers_num should be provided to determine the cluster number for each layer.


    Example
    -------
    >>> import torch
    >>> from speechbrain.lobes.models.huggingface_transformers.hubert import (HuBERT)
    >>> inputs = torch.rand([3, 2000])
    >>> model_hub = "facebook/hubert-large-ll60k"
    >>> save_path = "savedir"
    >>> ssl_layer_num = [7,23]
    >>> deduplicate =[False, True]
    >>> bpe_tokenizers=[None, None]
    >>> kmeans_repo_id = "speechbrain/SSL_Quantization"
    >>> kmeans_dataset = "LJSpeech"
    >>> num_clusters = 1000
    >>> ssl_model = HuBERT(model_hub, save_path,output_all_hiddens=True)
    >>> model = DiscreteSSL(save_path, ssl_model, kmeans_repo_id=kmeans_repo_id, kmeans_dataset=kmeans_dataset,num_clusters=num_clusters)
    >>> tokens, embs ,pr_tokens= model(inputs,SSL_layers=ssl_layer_num, deduplicates=deduplicate, bpe_tokenizers=bpe_tokenizers)
    >>> print(tokens.shape)
    torch.Size([3, 6, 2])
    >>> print(embs.shape)
    torch.Size([3, 6, 2, 1024])
    >>> print(pr_tokens.shape)
    torch.Size([3, 6, 2])
    """

    def __init__(
        self,
        ssl_model,
        kmeans_path,
        num_clusters=100,
        layers_num=None,
        Ktarget=512
    ):

        super().__init__()
        self.ssl_model = ssl_model
        # model_name = ssl_model.__class__.__name__.lower()
        self.check_if_input_is_compatible(layers_num, num_clusters)

        self.kmeans_models, self.ssl_layer_ids, self.num_clusters = (
            self.load_kmeans(
                kmeans_path=kmeans_path,
                num_clusters=num_clusters,
                layers_num=layers_num,
            )
        )

        self.vocabularies = []
        lin_tfs = []
        for model in self.kmeans_models:
            self.vocabularies.append(model.cluster_centers_)
            lin_tfs.append(nn.Linear(model.cluster_centers_.shape[1], Ktarget))
        self.lin_tfs = nn.ModuleList(lin_tfs)
        # self.tokenizer = DiscreteSSLTokenizer(self.num_clusters)
        



    def check_if_input_is_compatible(self, layers_num, num_clusters):
        """check if layer_number and num_clusters is consisntent with each other.
        Arguments
        ---------
        num_clusters:  int or List[int]
            determine the number of clusters of the targeted kmeans models to be downloaded. It could be varying for each layer.
        layers_num: : List[int] (Optional)
            If num_clusters is a list, the layers_num should be provided to determine the cluster number for each layer.
        """

        if layers_num:
            if isinstance(num_clusters, int):
                num_clusters = [num_clusters for i in layers_num]
            assert len(num_clusters) == len(
                layers_num
            ), "length of num_clusters and layers_num should be the same!!!"
        if layers_num is None:
            assert isinstance(
                num_clusters, int
            ), "num_clusters is expected to be int since the layers_num is not provided."
        self.num_clusters = num_clusters

    def load_kmeans(
        self,
        kmeans_path,
        num_clusters,
        cache_dir='.',
        layers_num=None,
    ):
        """Load a Pretrained kmeans model from HF.

        Arguments
        ---------
        repo_id : str
           The hugingface repo id that contains the model.
        kmeans_dataset : str
            Name of the dataset that Kmeans model are trained with in HF repo that need to be downloaded.
        cache_dir: str
            Path (dir) of the downloaded model.
        num_clusters:  int or List[int]
            determine the number of clusters of the targeted kmeans models to be downloaded. It could be varying for each layer.
        layers_num: : List[int] (Optional)
            If num_clusters is a list, the layers_num should be provided to determine the cluster number for each layer.
        Returns:
        ---------
        kmeans_model : MiniBatchKMeans:
            pretrained Kmeans  model loaded from the HF.
        layer_ids : List[int] :
            supported layer nums for kmeans (extracted from the name of kmeans model.)
        """

        kmeans_models = []
        layer_ids = []
        file_patterns = []
        if layers_num:
            for layer in layers_num:
                file_patterns.append(
                    f"*-{num_clusters}_layer_{layer}.pt"
                )
        else:
            file_patterns.append(
                f"*-{num_clusters[i]}.pt"
            )
        # kmeans_dir = snapshot_download(
        #     repo_id=repo_id, allow_patterns=file_patterns, cache_dir=cache_dir
        # )

        files = []
        for ext in file_patterns:
            for file in glob.glob(os.path.join(kmeans_path, ext)):
                if file not in files:
                    files.append(file)
                    layer_ids.append(
                        int(
                            file.split("/")[-1].split('.')[0][-1]
                        )
                    )
                    kmeans_models.append(joblib.load(file))

        assert (
            len(layer_ids) > 0
        ), f"There is no trained k-means model available for *_k{num_clusters[i]}_L*"

        if isinstance(num_clusters, int):
            num_clusters = [num_clusters for i in layer_ids]

        layer_ids, kmeans_models, num_clusters = zip(
            *sorted(zip(layer_ids, kmeans_models, num_clusters))
        )
        return kmeans_models, layer_ids, num_clusters

    def forward(
        self,
        h,
        wav_lens=None,
        SSL_layers=None,
        deduplicates=None,
        bpe_tokenizers=None,
    ):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        wav_len : tensor
            The relative length of the wav given in SpeechBrain format.
        SSL_layers: List[int]:
            determine which layers of SSL should be used to extract information.
        deduplicates: List[boolean]:
            determine to apply deduplication(remove duplicate subsequent tokens) on the tokens extracted for the corresponding layer.
        bpe_tokenizers: List[int]:
            determine to apply subwording on the tokens extracted for the corresponding layer if the sentencePiece tokenizer is trained for that layer.
        Returns:
        ---------
        tokens : torch.Tensor
            A (Batch x Seq x num_SSL_layers) tensor of audio tokens
        emb : torch.Tensor
            A (Batch x Seq x num_SSL_layers x embedding_dim ) cluster_centers embeddings for each tokens
        processed_tokens : torch.Tensor
            A (Batch x Seq x num_SSL_layers) tensor of audio tokens after applying deduplication and subwording if necessary.
        """

        if SSL_layers is None:
            SSL_layers = self.ssl_layer_ids
        # if deduplicates is None:
        #     deduplicates = [False] * len(SSL_layers)
        # if bpe_tokenizers is None:
        #     bpe_tokenizers = [None] * len(SSL_layers)

        # assert (
        #     len(deduplicates) == len(SSL_layers) == len(bpe_tokenizers)
        # ), "length of SSL_layers,deduplicates,bpe_tokenizers should be the same!!!"
        embeddings = []
        token_ids = []

        for layer in SSL_layers:
            if layer not in self.ssl_layer_ids:
                raise ValueError(
                    f"Layer {layer} is not among trained layers for k-means. Supported layers are: {self.ssl_layer_ids}."
                )
        
        lens = []
        with torch.no_grad():
            for layer_num, model, vocabulary in zip(self.ssl_layer_ids, self.kmeans_models, self.vocabularies):
                if layer_num not in SSL_layers:
                    continue
                h_reshaped = h[layer_num].permute(0, 2, 3, 1)

                tokens = model.predict(h_reshaped.reshape(-1, h_reshaped.shape[-1]).cpu())
                embs = vocabulary[tokens]
                embs_tensor = torch.tensor(embs.reshape(h_reshaped.shape[0], -1, embs.shape[-1]),
                                           dtype=torch.float,
                                           device=h_reshaped.device,
                                           )
                embeddings.append(self.lin_tfs[layer_num](embs_tensor))

                tokens_tensor = torch.tensor(tokens.reshape(h_reshaped.shape[0], -1),
                                             dtype=torch.long,
                                             device=h_reshaped.device,
                                            )
                token_ids.append(tokens_tensor)
                lens.append(tokens_tensor.shape[-1])

            max_len = max(lens)
            for layer_num in self.ssl_layer_ids:
                token_ids[layer_num] = F.pad(token_ids[layer_num], pad=(0, max_len - token_ids[layer_num].shape[-1]))
                embeddings[layer_num] = F.pad(embeddings[layer_num], pad=(0, 0, 0, max_len - embeddings[layer_num].shape[1]))

        tokens = torch.stack(token_ids, 2)
        discrete_embeddings = torch.stack(embeddings, 2)

        #processed_tokens = self.tokenizer.encode(
        #    org_tokens, SSL_layers, deduplicates, bpe_tokenizers
        #)
        return tokens, discrete_embeddings

if __name__ == "__main__":
    # This flag enables the built-in cuDNN auto-tuner
    # torch.backends.cudnn.benchmark = True

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Tensorboard logging
    if hparams["use_tensorboard"]:
        from speechbrain.utils.train_logger import TensorboardLogger

        hparams["tensorboard_train_logger"] = TensorboardLogger(
            hparams["tensorboard_logs_folder"]
        )

    run_on_main(
        prepare_esc50,
        kwargs={
            "data_folder": hparams["data_folder"],
            "audio_data_folder": hparams["audio_data_folder"],
            "save_json_train": hparams["train_annotation"],
            "save_json_valid": hparams["valid_annotation"],
            "save_json_test": hparams["test_annotation"],
            "train_fold_nums": hparams["train_fold_nums"],
            "valid_fold_nums": hparams["valid_fold_nums"],
            "test_fold_nums": hparams["test_fold_nums"],
            "skip_manifest_creation": hparams["skip_manifest_creation"],
        },
    )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    datasets, label_encoder = dataio_prep(hparams)
    hparams["label_encoder"] = label_encoder

    if "wham_folder" in hparams:
        hparams["wham_dataset"] = prepare_wham(
            hparams["wham_folder"],
            hparams["add_wham_noise"],
            hparams["sample_rate"],
            hparams["signal_length_s"],
            hparams["wham_audio_folder"],
        )

    if hparams["wham_dataset"] is not None:
        assert hparams["signal_length_s"] == 5, "Fix wham sig length!"

    class_labels = list(label_encoder.ind2lab.values())
    print("Class Labels:", class_labels)

    ESC50_brain = ESC50Brain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Load pretrained encoder if it exists in the yaml file
    if not hasattr(ESC50_brain.modules, "embedding_model"):
        ESC50_brain.hparams.embedding_model.to(ESC50_brain.device)

    if "pretrained_encoder" in hparams and hparams["use_pretrained"]:
        run_on_main(hparams["pretrained_encoder"].collect_files)
        hparams["pretrained_encoder"].load_collected()

    if not hparams["test_only"]:
        ESC50_brain.fit(
            epoch_counter=ESC50_brain.hparams.epoch_counter,
            train_set=datasets["train"],
            valid_set=datasets["valid"],
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["dataloader_options"],
        )

    # Load the best checkpoint for evaluation
    test_stats = ESC50_brain.evaluate(
        test_set=datasets["test"],
        min_key="error",
        progressbar=True,
        test_loader_kwargs=hparams["dataloader_options"],
    )
