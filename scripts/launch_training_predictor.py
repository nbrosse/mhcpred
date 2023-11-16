from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from mhcpred.class1_binary_predictor import Class1BinaryPredictor
from mhcpred.config import settings
from mhcpred.data import get_train_data, get_test_data

data_path = Path(settings.data_path)


def run():
    allele_sequences = pd.read_csv(str(data_path / "allele_sequences.csv"), index_col=0).iloc[:, 0]
    df_total_train = get_train_data()
    df_test = get_test_data()
    alleles_in_use = set(df_total_train.allele).union(set(df_test.allele))

    allele_sequences_in_use = allele_sequences[allele_sequences.index.isin(alleles_in_use)]

    class1_binary_predictor = Class1BinaryPredictor(
        allele_to_sequence=allele_sequences_in_use.to_dict(),
    )

    df_train, df_val = train_test_split(df_total_train, test_size=0.1, shuffle=True, stratify=df_total_train.hit.values)

    class1_binary_predictor.fit(
        df_train=df_train,
        df_val=df_val,
        epochs=2,
    )


if __name__ == "__main__":
    run()
    # Process finished with exit code 138 (interrupted by signal 10: SIGBUS)