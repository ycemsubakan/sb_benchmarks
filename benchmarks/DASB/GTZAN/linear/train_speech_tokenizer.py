#!/usr/bin/env python3
"""Recipe for training asic  music genra classification using the gtzan dataset.
# This recipe uses the avg pooling + linear backbone for classification.

Authors
 * Pooneh Mousavi 2024
"""

import os
import torch
import torchaudio
import sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml


class MusGenreBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + emotion classifier."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        # Feature extraction and attention pooling
        with torch.no_grad():
            self.hparams.codec.to(self.device).eval()
            tokens = self.hparams.codec(wavs).permute(1, 2, 0)[
                :, :, : self.hparams.num_codebooks
            ]
        embeddings = self.modules.discrete_embedding_layer(tokens)
        att_w = self.modules.attention_mlp(embeddings)
        feats = torch.matmul(att_w.transpose(2, -1), embeddings).squeeze(-2)
  
        # last dim will be used for AdaptativeAVG pool
        outputs = self.hparams.avg_pool(feats, wav_lens)
        outputs = outputs.view(outputs.shape[0], -1)
        outputs = self.modules.classifier(outputs)
        outputs = self.hparams.log_softmax(outputs)
        return outputs

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label."""
        genreid, _ = batch.genre_encoded

        """to meet the input form of nll loss"""
        genreid = genreid.squeeze(1)
        loss = self.hparams.compute_cost(predictions, genreid)
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, genreid)
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

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "error_rate": self.error_metrics.summarize("average"),
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing_model(
                stats["error_rate"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.model_optimizer, new_lr
            )

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(
                meta=stats, min_keys=["error_rate"]
            )

        # We also write statistics about test data to stdout and to logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

    def init_optimizers(self):
        "Initializes the weights optimizer and model optimizer"
        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )
        self.optimizers_dict = {
            "model_optimizer": self.model_optimizer,
        }
        # Initializing the weights
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)


def dataio_prep(hparams):
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
            info.sample_rate, hparams["sample_rate"],
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
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "genre_encoded"],
        )
    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mappinng.

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="genre",
    )

    return datasets


# RECIPE BEGINS!
if __name__ == "__main__":
    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    from gtzan_prepare import prepare_gtzan  # noqa E402

    # Data preparation, to be run on only one process.
    if not hparams["skip_prep"]:
        sb.utils.distributed.run_on_main(
            prepare_gtzan,
            kwargs={
                "data_folder": hparams["data_folder"],
                "save_json_train": hparams["train_annotation"],
                "save_json_valid": hparams["valid_annotation"],
                "save_json_test": hparams["test_annotation"],
            },
        )

    # Data preparation, to be run on only one process.
    # Create dataset objects "train", "valid", and "test".
    datasets = dataio_prep(hparams)

    # Initialize the Brain object to prepare for mask training.
    mus_genre_brain = MusGenreBrain(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    mus_genre_brain.fit(
        epoch_counter=mus_genre_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
    # Load the best checkpoint for evaluation
    test_stats = mus_genre_brain.evaluate(
        test_set=datasets["test"],
        min_key="error_rate",
        test_loader_kwargs=hparams["dataloader_options"],
    )
