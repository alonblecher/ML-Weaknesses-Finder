import pymc3
import pandas as pd
import numpy as np


def hdr(df_, Y, column, eps = 0.05, threshold = 0.01):
  df = df_.copy()
  df['is_correct_pred'] = Y
  
  range_diff_history = []
  start, end = pymc3.stats.hdi(df[column].values, hdi_prob=0.5)

  data_interval = df[df[column].between(start, end)]
  error_rate = (data_interval['is_correct_pred'] == False).sum() / df.shape[0]

  while (df[df[column].between(start, end)].shape[0] / df.shape[0] > 0.1):

    prev_start = start
    prev_end = end

    start = start * (1 + eps)
    end = end * (1 - eps)

    data_interval = df[df[column].between(start, end)]
    new_error_rate = (data_interval['is_correct_pred'] == False).sum() / df.shape[0]

    if (error_rate - new_error_rate > threshold):
      range_diff = {
          'error_rate' : error_rate - new_error_rate,
          'range_start' : (prev_start, start),
          'range_end': (end, prev_end)
      }
      range_diff_history.append(range_diff)
    error_rate = new_error_rate
  N = int(0.25 * len(range_diff_history))
  n_largest_diffs = sorted(range_diff_history, key=lambda t: t['error_rate'], reverse=True)[:N]

  return n_largest_diffs

def hdr_ranges_to_slice(df, column, ranges):
  df_slices = []
  for range in ranges:
    df_range = df[df[column].between(range['range_start'][0], range['range_start'][1]) | df[column].between(range['range_end'][0], range['range_end'][1])]
    df_slices.append(df_range)
  return pd.concat(df_slices)
