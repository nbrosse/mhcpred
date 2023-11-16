import pickle
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from mhcflurry.allele_encoding import AlleleEncoding
from mhcflurry.encodable_sequences import EncodableSequences
from sklearn.model_selection import train_test_split

from mhcpred.config import settings
from mhcpred.data import get_train_data, get_test_data
from mhcpred.hyperparameters import base_hyperparameters
from mhcpred.class1_binary_nn import Class1BinaryNeuralNetwork

data_path = Path(settings.data_path)
models_path = Path(settings.models_path)
output_path = Path(settings.output_path)


allele_sequences = pd.read_csv(str(data_path / "allele_sequences.csv"), index_col=0).iloc[:, 0]

df_total_train = get_train_data()
df_test = get_test_data()
alleles_in_use = set(df_total_train.allele).union(set(df_test.allele))

allele_sequences_in_use = allele_sequences[allele_sequences.index.isin(alleles_in_use)]

allele_encoding = AlleleEncoding(
    alleles=allele_sequences_in_use.index.values,
    allele_to_sequence=allele_sequences_in_use.to_dict()
)

df_train, df_val = train_test_split(df_total_train, test_size=0.1, shuffle=True, stratify=df_total_train.hit.values)

val_peptides = EncodableSequences(df_val.peptide.values)
val_alleles = AlleleEncoding(
    alleles=df_val.allele.values,
    allele_to_sequence=allele_sequences_in_use.to_dict(),
)

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
            allele_encoding = AlleleEncoding(
                alleles=df.allele.values,
                borrow_from=train_allele_encoding,
            )
            yield (allele_encoding, encodable_peptides, df.hit.values)


batch_size = 1024

train_generator = train_data_iterator(
    df_train=df_train,
    train_allele_encoding=allele_encoding,
    batch_size=batch_size,
)

model = Class1BinaryNeuralNetwork(**base_hyperparameters)

steps_per_epoch = np.ceil(len(df_train) / batch_size)

model.fit_generator(
    generator=train_generator,
    validation_peptide_encoding=val_peptides,
    validation_affinities=df_val.hit.values,
    validation_allele_encoding=val_alleles,
    validation_inequalities=None,
    validation_output_indices=None,
    steps_per_epoch=steps_per_epoch,
    epochs=2,
)

with open(str(models_path / "model.pickle"), "wb") as f:
    pickle.dump(model, f)

test_peptides = df_test.peptide.values
test_allele_encoding = AlleleEncoding(
    alleles=df_test.allele.values,
    allele_to_sequence=allele_sequences_in_use.to_dict(),
)

predictions = model.predict(
    peptides=test_peptides,
    allele_encoding=test_allele_encoding,
)

df_test["predictions"] = predictions
df_test.to_csv(str(output_path / "mhcpred_predictions.csv"), index=False)