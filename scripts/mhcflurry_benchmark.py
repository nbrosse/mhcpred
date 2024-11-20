from pathlib import Path

import pandas as pd
from mhcflurry import Class1AffinityPredictor

from src.mhcpred.config import settings
from src.mhcpred.data import get_test_data

output_path = Path(settings.output_path)


def predict_with_mhcflurry() -> pd.DataFrame:
    predictor = Class1AffinityPredictor.load()
    df_test = get_test_data()
    mhcflurry_predictions = predictor.predict_to_dataframe(
        peptides=df_test.peptide.values,
        alleles=df_test.allele.values,
        allele=None,
    )
    df = pd.merge(df_test, mhcflurry_predictions, on=["allele", "peptide"], how="left")
    df.to_csv(str(output_path / "mhcflurry_predictions.csv"), index=False)
    return df


if __name__ == "__main__":
    predict_with_mhcflurry()
