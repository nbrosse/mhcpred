import pickle
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from mhcflurry.allele_encoding import AlleleEncoding
from mhcflurry.encodable_sequences import EncodableSequences
from sklearn.model_selection import train_test_split

from mhcpred.class1_binary_nn import Class1BinaryNeuralNetwork
from mhcpred.config import settings
from mhcpred.data import get_test_data, get_train_data
from mhcpred.hyperparameters import base_hyperparameters

# Define paths for data, models, and output using settings.
data_path = Path(settings.data_path)
models_path = Path(settings.models_path)
output_path = Path(settings.output_path)

# Load allele sequences from a CSV file. The index is set to the allele name.
allele_sequences = pd.read_csv(
    str(data_path / "allele_sequences.csv"), index_col=0
).iloc[:, 0]

# Load training and test data using helper functions.
df_total_train = get_train_data()
df_test = get_test_data()

# Identify the set of alleles present in both training and test data.
alleles_in_use = set(df_total_train.allele).union(set(df_test.allele))

# Filter allele sequences to only include those present in the training and test sets.
allele_sequences_in_use = allele_sequences[allele_sequences.index.isin(alleles_in_use)]

# Create an AlleleEncoding object to convert allele names to numerical representations.
allele_encoding = AlleleEncoding(
    alleles=allele_sequences_in_use.index.values,
    allele_to_sequence=allele_sequences_in_use.to_dict(),
)

# Split the training data into training and validation sets. Stratified splitting
# ensures that the proportion of positive and negative samples (hit column) is
# maintained across the splits.
df_train, df_val = train_test_split(
    df_total_train, test_size=0.1, shuffle=True, stratify=df_total_train.hit.values
)

# Prepare validation data. Encode peptides and alleles.
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
    """
    This function creates a data generator for training the neural network.
    It iterates over the training data in batches and yields tuples of
    (allele_encoding, peptide_sequences, labels).  It also handles filtering
    of alleles not found in the initial allele encoding.
    """
    # Get unique alleles in the training set.
    alleles = df_train.allele.unique()
    # Filter alleles to keep only those for which sequences are available.
    usable_alleles = [
        c for c in alleles if c in train_allele_encoding.allele_to_sequence
    ]
    print("Using %d / %d alleles" % (len(usable_alleles), len(alleles)))
    print(
        "Skipped alleles: ",
        [c for c in alleles if c not in train_allele_encoding.allele_to_sequence],
    )
    df_train = df_train.query("allele in @usable_alleles")

    # Calculate the number of batches.
    n_splits = np.ceil(len(df_train) / batch_size)

    # Infinite loop to allow for multiple epochs.
    while True:
        # Split the training data into batches.
        epoch_dfs = np.array_split(df_train.copy(), n_splits)
        for k, df in enumerate(epoch_dfs):
            if len(df) == 0:
                continue
            # Encode peptides and alleles for the current batch.
            encodable_peptides = EncodableSequences(df.peptide.values)
            allele_encoding = AlleleEncoding(
                alleles=df.allele.values,
                borrow_from=train_allele_encoding,  # Reuse encoding from main allele_encoding
            )
            # Yield the encoded data and labels (hit column).
            yield (allele_encoding, encodable_peptides, df.hit.values)


# Set the batch size for training.
batch_size = 1024

# Create the training data generator.
train_generator = train_data_iterator(
    df_train=df_train,
    train_allele_encoding=allele_encoding,
    batch_size=batch_size,
)

# Initialize the neural network model with base hyperparameters.
model = Class1BinaryNeuralNetwork(**base_hyperparameters)

# Calculate the number of training steps per epoch.
steps_per_epoch = np.ceil(len(df_train) / batch_size)

# Train the model using the generator.
model.fit_generator(
    generator=train_generator,
    validation_peptide_encoding=val_peptides,
    validation_affinities=df_val.hit.values,
    validation_allele_encoding=val_alleles,
    validation_inequalities=None, # Not used in this example
    validation_output_indices=None, # Not used in this example
    steps_per_epoch=steps_per_epoch,
    epochs=2, # Number of training epochs
)

# Save the trained model to a pickle file.
with open(str(models_path / "model.pickle"), "wb") as f:
    pickle.dump(model, f)

# Prepare test data.
test_peptides = df_test.peptide.values
test_allele_encoding = AlleleEncoding(
    alleles=df_test.allele.values,
    allele_to_sequence=allele_sequences_in_use.to_dict(),
)

# Make predictions on the test data.
predictions = model.predict(
    peptides=test_peptides,
    allele_encoding=test_allele_encoding,
)

# Add predictions to the test dataframe and save it to a CSV file.
df_test["predictions"] = predictions
df_test.to_csv(str(output_path / "mhcpred_predictions.csv"), index=False)