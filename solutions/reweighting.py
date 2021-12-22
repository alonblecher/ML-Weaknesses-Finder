import numpy as np
from xgboost import XGBClassifier


def reweighting(train_df, target_column, indexes, weight = 5):
  sample_weights = np.ones(train_df.shape[0]) 
  sample_weights[indexes] = weight

  clf = XGBClassifier()
  clf.fit(train_df.drop(columns = [target_column], axis = 0), train_df[target_column], sample_weight = sample_weights)

  return clf