from pathlib import Path

import numpy as np
import pandas as pd
from mhcflurry.common import normalize_allele_name

from src.mhcpred.config import settings

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
