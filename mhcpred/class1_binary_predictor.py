import collections
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from mhcflurry import Class1AffinityPredictor
from mhcflurry.allele_encoding import AlleleEncoding
from mhcflurry.encodable_sequences import EncodableSequences

from mhcpred.class1_binary_nn import Class1BinaryNeuralNetwork
from mhcpred.config import settings
from mhcpred.hyperparameters import base_hyperparameters

data_path = Path(settings.data_path)


def train_data_iterator(
        df_train: pd.DataFrame,
        train_allele_encoding: AlleleEncoding,
        batch_size: int = 1024,
) -> Iterator[tuple[AlleleEncoding, EncodableSequences, np.ndarray]]:

    # Supported alleles and filtering
    alleles = df_train.allele.unique()
    usable_alleles = [
        c for c in alleles
        if c in train_allele_encoding.allele_to_sequence
    ]
    print("Using %d / %d alleles" % (len(usable_alleles), len(alleles)))
    print("Skipped alleles: ", [
        c for c in alleles
        if c not in train_allele_encoding.allele_to_sequence
    ])
    df_train = df_train.query("allele in @usable_alleles")

    # Divide into batches
    n_splits = np.ceil(len(df_train) / batch_size)

    while True:
        epoch_dfs = np.array_split(df_train.copy(), n_splits)
        for (k, df) in enumerate(epoch_dfs):
            if len(df) == 0:
                continue
            encodable_peptides = EncodableSequences(df.peptide.values)
            train_allele_encoding = AlleleEncoding(
                alleles=df.allele.values,
                borrow_from=train_allele_encoding,
            )
            yield (train_allele_encoding, encodable_peptides, df.hit.values)


class Class1BinaryPredictor(Class1AffinityPredictor):

    def __init__(self, allele_to_sequence=None) -> None:
        super(Class1BinaryPredictor, self).__init__(
            allele_to_allele_specific_models=None,
            class1_pan_allele_models=None,
            allele_to_sequence=allele_to_sequence,
            manifest_df=None,
            allele_to_percent_rank_transform=None,
            metadata_dataframes=None,
            provenance_string=None,
            optimization_info=None,
        )

    def fit(
            self,
            df_train: pd.DataFrame,
            df_val: pd.DataFrame,
            architecture_hyperparameters=base_hyperparameters,
            models_dir_for_save=settings.models_path,
            batch_size: int = 1024,
            epochs=5,
            min_epochs=0,
            patience=10,
            min_delta=0.0,
            verbose=1,
            progress_preamble="",
            progress_print_interval=5.0):

        # val
        val_allele_encoding = AlleleEncoding(
            df_val.allele.values,
            allele_to_sequence=dict(self.allele_to_sequence),
        )
        val_peptides = EncodableSequences(df_val.peptide.values)

        # train iterator
        train_allele_encoding = AlleleEncoding(
            allele_to_sequence=dict(self.allele_to_sequence),
        )
        train_generator = train_data_iterator(
            df_train=df_train,
            train_allele_encoding=train_allele_encoding,
            batch_size=batch_size,
        )
        steps_per_epoch = np.ceil(len(df_train) / batch_size)

        model = Class1BinaryNeuralNetwork(**architecture_hyperparameters)
        model.fit_generator(
            generator=train_generator,
            validation_peptide_encoding=val_peptides,
            validation_affinities=df_val.hit.values,
            validation_allele_encoding=val_allele_encoding,
            validation_inequalities=None,
            validation_output_indices=None,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            min_epochs=min_epochs,
            patience=patience,
            min_delta=min_delta,
            verbose=verbose,
            progress_preamble=progress_preamble,
            progress_print_interval=progress_print_interval,
        )

        model_name = self.model_name("pan-class1", 0)
        row = pd.Series(collections.OrderedDict([
            ("model_name", model_name),
            ("allele", "pan-class1"),
            ("config_json", json.dumps(model.get_config())),
            ("model", model),
        ])).to_frame().T
        self._manifest_df = pd.concat(
            [self.manifest_df, row], ignore_index=True)
        self.class1_pan_allele_models.append(model)
        if models_dir_for_save:
            self.save(models_dir_for_save, model_names_to_write=[model_name])

        self.clear_cache()
        return model


