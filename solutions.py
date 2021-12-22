import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
from ydata_synthetic.synthesizers.regular import WGAN_GP
from detectors.hdr import hdr, hdr_ranges_to_slice
from detectors.decision_tree import tree_indexes_by_slices


def reweighting(train_df, target_column, indexes, weight = 5):
  sample_weights = np.ones(train_df.shape[0]) 
  sample_weights[indexes] = weight

  clf = XGBClassifier()
  clf.fit(train_df.drop(columns = [target_column], axis = 0), train_df[target_column], sample_weight = sample_weights)

  return clf

def synthesized_data(train_samples, generator_sample_size):

  #config
  noise_dim = 32
  dim = 256
  batch_size = 256 if 256 <= train_samples.shape[0] else train_samples.shape[0]
  
  #train config
  log_step = 20
  epochs = 50+1
  learning_rate = 5e-4
  beta_1 = 0.5
  beta_2 = 0.9

  gan_args = ModelParameters(batch_size=batch_size,
                           lr=learning_rate,
                           betas=(beta_1, beta_2),
                           noise_dim=noise_dim,
                           n_cols=train_samples.shape[1],
                           layers_dim=dim)
  train_args = TrainParameters(epochs=epochs,
                             sample_interval=log_step)
  # Train GAN
  model = WGAN_GP
  synthesizer = model(gan_args, n_critic=5)
  synthesizer.train(train_samples, train_args)

  # Generate records based on random noise
  generator = synthesizer.generator
  rand_noise = np.random.normal(size=(generator_sample_size, noise_dim))
  generated_samples = generator.predict(rand_noise)
  return generated_samples

def apply_synthesized_data(train_df, target_column, indexes, options, data = False):
  train_df_drop_target = train_df.drop(columns = [target_column])
  train_samples = train_df[train_df.index.isin(indexes)]

  train_samples_positive = train_samples[train_samples[target_column] == 1].drop(columns = [target_column])
  train_samples_negative = train_samples[train_samples[target_column] == 0].drop(columns = [target_column])

  synthesized_positive = synthesized_data(train_samples_positive, options.get('generator_sample_size' ,train_samples_positive.shape[0] * 2))
  synthesized_negative = synthesized_data(train_samples_negative, options.get('generator_sample_size' ,train_samples_negative.shape[0] * 2))

  synthesized_positive_df = pd.DataFrame(synthesized_positive, columns = train_df_drop_target.columns)
  synthesized_negative_df = pd.DataFrame(synthesized_negative, columns = train_df_drop_target.columns)

  synthesized_positive_df[target_column] = np.ones(synthesized_positive_df.shape[0])
  synthesized_negative_df[target_column] = np.zeros(synthesized_negative_df.shape[0])

  if data:
    return pd.concat([train_samples, synthesized_positive_df, synthesized_negative_df])

  df_combined = pd.concat([train_df, synthesized_positive_df, synthesized_negative_df])

  clf = XGBClassifier()
  clf.fit(df_combined.drop(columns = [target_column]), df_combined[target_column])

  return clf

def ad_hoc_model(train_df, target_column, indexes, options):

  train_samples = apply_synthesized_data(train_df, target_column, indexes, options, True)

  clf = XGBClassifier()
  clf.fit(train_samples.drop(columns = [target_column]), train_samples[target_column])

  return clf