from pathlib import Path

import pandas as pd
from mhcflurry import Class1AffinityPredictor
from sklearn.metrics import accuracy_score

from mhcpred.config import settings
from mhcpred.data import get_test_data

# predictor = Class1AffinityPredictor.load()

df_test = get_test_data()

# mhcflurry_predictions = predictor.predict_to_dataframe(
#     peptides=df_test.peptide.values,
#     alleles=df_test.allele.values,
#     allele=None,
# )

data_path = Path(settings.data_path)

mhcflurry_predictions = pd.read_csv(str(data_path / "mhcflurry_predictions.csv"))
df = pd.merge(df_test, mhcflurry_predictions, on=["allele", "peptide"], how="inner")
# assert len(df) == len(df_test)

# df_head = df.head(1000)
# del df

y_pred = df.prediction_percentile.values <= 2
score = accuracy_score(y_true=df.hit.values, y_pred=y_pred)
