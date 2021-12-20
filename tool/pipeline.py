import pandas as pd
import numpy as np
from tool.detectors.hdr import hdr, hdr_ranges_to_slice
from tool.detectors.decision_tree import get_decision_tree_slices, tree_indexes_by_slices
from heapq import nlargest


def initalize_data_set(target_column, predicted_column, categorical_threshold = 0.001, csv_file_path = None , df = None):
  if csv_file_path is None and df is None:
    raise Exception("expected csv file path or data frame object")
  if df is None:
    df = pd.read_csv(csv_file_path)
  else:
    if not isinstance(df, pd.DataFrame):
      raise Exception("data frame object must be of type 'pandas.core.frame.DataFrame'")
      
  if predicted_column == "":
    X = df.drop([target_column], axis=1)
    Y = df[target_column]
  else:
    X = df.drop([target_column, predicted_column], axis=1)
    Y = (df[predicted_column] == df[target_column])

  categorical_features = {}
  for feature in X.columns:
      categorical_features[feature] = 1.*X[feature].nunique()/X[feature].count() < categorical_threshold or X[feature].dtype == "object"

  X_encoded = pd.get_dummies(X, columns= [key for (key, value) in categorical_features.items() if value ])

  return X_encoded, Y, categorical_features


def get_all_indexes_from_all_slices(df, slices):
  categorical_slices = []
  continious_slices = []
  for key in slices.keys():
    # for categorical
    if 'range' in slices[key][0]:
      categorical_slices.append(slices[key])
    else:
      continious_slices.append(hdr_ranges_to_slice(df, key, slices[key]))

  df_categorical = tree_indexes_by_slices(df, categorical_slices)
  df_continious = None
  if len(continious_slices) > 0:
    df_continious = pd.concat(continious_slices)
  if len(continious_slices) > 0 and len(df_categorical) > 0:
    return pd.concat([df_categorical, df_continious]).index.unique()
  elif len(df_categorical) == 0:
    return df_continious.index.unique()
  else:
    return df_categorical.index.unique()

def apply_heuristics(X, Y, df, features, options = {}):
  high_rate_columns = [column for column in df.columns if df[column].value_counts().max() / df.shape[0] > 0.7 ]

  categorical_features = [key for (key, value) in features.items() if (value and (key not in high_rate_columns))]
  continious_features = [key for (key, value) in features.items() if ((value == False) and (key not in high_rate_columns))]
  #categorical_features
  categorical_features_error_rates_single = {}
  categorical_features_slices_single = {}
  for feature in categorical_features:
    cols = [c for c in X.columns if f'{feature}_' in c]
    slices, graph = get_decision_tree_slices(X, Y, cols)
    if len(slices) > 0:
      categorical_features_error_rates_single[feature] = np.mean([slice_dict['error_rate'] for slice_dict in slices])
      categorical_features_slices_single[feature] = slices
    
  #continious_features
  continious_feature_error_rates_single = {}
  continious_feature_slices_single = {}
  for feature in continious_features:
    slices = hdr(X, Y, feature, options.get('eps', 0.05), options.get('hdr_threshold', 0.001)) 
    if len(slices) > 0:
      continious_feature_error_rates_single[feature] = np.mean([slice_dict['error_rate'] for slice_dict in slices])
      continious_feature_slices_single[feature] = slices

  #combined top quarter
  combined_features_slices_single = categorical_features_slices_single.copy()
  combined_features_slices_single.update(continious_feature_slices_single)
  combined_features_error_rates_single = categorical_features_error_rates_single.copy()
  combined_features_error_rates_single.update(continious_feature_error_rates_single)
  N = int(0.25 * len(combined_features_error_rates_single.keys()))
  largest_feature_error_rates_single = nlargest(N, combined_features_error_rates_single, key = combined_features_error_rates_single.get)
  
  #feature pairs
  filtered_features = [x for x in features if (x not in set(largest_feature_error_rates_single) and x not in high_rate_columns)]
  feature_error_rates_pairs = {}
  feature_slices_pairs = {}
  for feature_largest in largest_feature_error_rates_single:
    for feature in filtered_features:
      cols = [c for c in X.columns if f'{feature_largest}_' in c or f'{feature}_' in c]
      slices, graph = get_decision_tree_slices(X, Y, cols)

      if len([slice_dict['error_rate'] for slice_dict in slices])!=0:
        feature_error_rates_pairs[f'{feature_largest}_{feature}'] = np.mean([slice_dict['error_rate'] for slice_dict in slices])
        feature_slices_pairs[f'{feature_largest}_{feature}'] = slices

  N = int(0.25 * len(feature_error_rates_pairs.keys()))
  largest_feature_error_rates_pairs = nlargest(N, feature_error_rates_pairs, key = feature_error_rates_pairs.get)

  combined_features_slices_single.update(feature_slices_pairs)
  combined_features_error_rates_single.update(feature_error_rates_pairs)
  N = int(0.25 * len(combined_features_slices_single.keys()))
  largest_feature_error_rates_combined = nlargest(N, combined_features_error_rates_single, key = combined_features_error_rates_single.get)

  top_features_slices = {field:max(slice_list, key=lambda x:x['error_rate']) for (field,slice_list) in combined_features_slices_single.items()}

  return top_features_slices, largest_feature_error_rates_combined