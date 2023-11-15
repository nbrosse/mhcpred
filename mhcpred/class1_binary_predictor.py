import collections
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from mhcflurry import Class1AffinityPredictor
from mhcflurry.allele_encoding import AlleleEncoding
from mhcflurry.encodable_sequences import EncodableSequences

from mhcpred.class1_binary_nn import Class1BinaryNeuralNetwork
from mhcpred.config import settings
from mhcpred.data import train_data_iterator
from mhcpred.hyperparameters import base_hyperparameters

data_path = Path(settings.data_path)


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
            models_dir_for_save=str(data_path / "models"),
            batch_size: int = 2014,
            epochs=1000,
            min_epochs=0,
            patience=10,
            min_delta=0.0,
            verbose=1,
            progress_preamble="",
            progress_print_interval=5.0):

        # val
        val_allele_encoding = AlleleEncoding(
            df_val.allele.values,
            borrow_from=self.master_allele_encoding,
        )
        val_peptides = EncodableSequences(df_val.peptide.values)

        # train iterator
        train_generator = train_data_iterator(
            df_train=df_train,
            allele_encoding=self.master_allele_encoding,
            batch_size=batch_size,
        )
        steps_per_epoch = np.ceil(len(df_train) / batch_size)

        logging.info("Training model")
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