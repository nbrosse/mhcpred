from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from mhcflurry.allele_encoding import AlleleEncoding
from mhcflurry.common import normalize_allele_name
from mhcflurry.encodable_sequences import EncodableSequences
from sklearn.model_selection import train_test_split

from mhcpred.config import settings

data_path = Path(settings.data_path)
netmhcpan41_data_path = data_path / "netmhcpan41_data"


def get_train_data() -> pd.DataFrame:
    """
    columns = Index(['peptide', 'allele', 'hit', 'fold'], dtype='object')
    """
    dfs = list()
    for i in range(5):
        df = pd.read_csv(str(netmhcpan41_data_path / f"fold_{i}.csv"))
        df["fold"] = i
        dfs.append(df)
    df = pd.concat(dfs, axis=0).reset_index(drop=True)
    df = df.astype({"hit": bool, "fold": np.uint8})
    return df


def get_test_data() -> pd.DataFrame:
    df = pd.read_csv(str(netmhcpan41_data_path / "test.csv"))
    df = df.astype({"hit": bool})
    df.loc[:, "allele"] = df.allele.map(normalize_allele_name)
    return df


def train_data_iterator(
        df_train: pd.DataFrame,
        allele_encoding: AlleleEncoding,
        batch_size: int = 1024,
) -> Iterator[tuple[AlleleEncoding, EncodableSequences, np.ndarray]]:

    # Supported alleles and filtering
    alleles = df_train.allele.unique()
    usable_alleles = [
        c for c in alleles
        if c in allele_encoding.allele_to_sequence
    ]
    print("Using %d / %d alleles" % (len(usable_alleles), len(alleles)))
    print("Skipped alleles: ", [
        c for c in alleles
        if c not in allele_encoding.allele_to_sequence
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
                borrow_from=allele_encoding,
            )
            yield (allele_encoding, encodable_peptides, df.hit.values)


# df_train = get_train_data()
# df_test = get_test_data()

# df_1, df_2 = train_test_split(df_train, test_size=0.1, shuffle=True, stratify=df_train.hit.values)

# allele_sequences = pd.read_csv(str(data_path / "allele_sequences.csv"), index_col=0).iloc[:, 0]
# assert set(df_train.allele.unique()) <= set(allele_sequences.index)
# assert set(df_test.allele.unique()) <= set(allele_sequences.index)