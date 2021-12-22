from xgboost import XGBClassifier
from solutions.sythesized_data import apply_synthesized_data

def ad_hoc_model(train_df, target_column, indexes, options):

  train_samples = apply_synthesized_data(train_df, target_column, indexes, options, True)

  clf = XGBClassifier()
  clf.fit(train_samples.drop(columns = [target_column]), train_samples[target_column])

  return clf