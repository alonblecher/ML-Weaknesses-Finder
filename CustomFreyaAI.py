import pandas as pd
from xgboost import XGBClassifier
from pipeline import initalize_data_set, apply_heuristics, get_all_indexes_from_all_slices
from sklearn.model_selection import train_test_split
from sklearn import metrics
from solutions import reweighting, ad_hoc_model, apply_synthesized_data

class CustomFreyaAI:
  def __init__(self, df, target_column = 'y'):
    if not isinstance(df, pd.DataFrame):
      raise Exception("data frame object must be of type 'pandas.core.frame.DataFrame'")
    if not target_column in df:
      raise Exception(f"The specified target column '{target_column}', was not found in the data frame")
    self.df = df.copy()
    self.target_column = target_column

  def get_slices_report(self, options = {}):
    X_encoded, Y, categorical_features = initalize_data_set(target_column = self.target_column, predicted_column = '',df = self.df, categorical_threshold = options.get('categorical_threshold', 0.001))

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, Y, test_size=0.2, random_state=2, stratify=Y)

    self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

    clf = XGBClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    feature_slices, largest_feature_error_rates  = apply_heuristics(X_test, y_pred == y_test, self.df, categorical_features, options)

    self.feature_slices = feature_slices
    return feature_slices, largest_feature_error_rates

  def apply_solution(self, feature_name, solution_name, options = {}):
    if not feature_name in self.feature_slices:
        raise Exception(f"The specified feature name '{feature_name}', was not found among the features")
    if not solution_name in ["reweighting", "data_synthesizer", "Ad_Hoc"]:
        raise Exception(f"The specified solution name '{solution_name}', was not found among the available solutions")

    print(f"Applying {solution_name}...")

    report = {}

    X_train_copy = self.X_train.copy()
    X_train_copy[self.target_column] = self.y_train
    X_train_copy.reset_index(inplace = True)
    X_train_copy.drop(columns = ['index'], inplace = True)

    train_indexes = get_all_indexes_from_all_slices(X_train_copy, {feature_name:[self.feature_slices[feature_name]]})
    test_indexes = get_all_indexes_from_all_slices(self.X_test, {feature_name:[self.feature_slices[feature_name]]})

    clf = XGBClassifier()
    clf.fit(self.X_train, self.y_train)
    pred = clf.predict(self.X_test[self.X_test.index.isin(test_indexes)])
    before_slice_score = metrics.accuracy_score(self.y_test[self.y_test.index.isin(test_indexes)], pred)

    print(f'Slice size to overall size: {train_indexes.shape[0]}/{X_train_copy.shape[0]} = {train_indexes.shape[0]/X_train_copy.shape[0]}')
    report['before_slice_score'] = before_slice_score

    pred = clf.predict(self.X_test)
    before_overall_score = metrics.accuracy_score(self.y_test, pred)
    report['before_overall_score'] = before_overall_score

    if solution_name == 'reweighting':
      clf = reweighting(X_train_copy, self.target_column, train_indexes, options.get('weight', 5))
    elif solution_name == 'data_synthesizer':
      clf = apply_synthesized_data(X_train_copy, self.target_column, train_indexes, options)
    elif solution_name == 'Ad_Hoc':
      clf = ad_hoc_model(X_train_copy, self.target_column, train_indexes, options)

    pred = clf.predict(self.X_test[self.X_test.index.isin(test_indexes)])
    after_slice_score = metrics.accuracy_score(self.y_test[self.y_test.index.isin(test_indexes)], pred)
    report['after_slice_score'] = after_slice_score

    after_overall_score = None
    
    if solution_name == 'Ad_Hoc':
      after_overall_score = before_overall_score
    else:
      pred = clf.predict(self.X_test)
      after_overall_score = metrics.accuracy_score(self.y_test, pred)

    report['after_overall_score'] = after_overall_score

    report['after_to_before_slice_performance_ratio'] = after_slice_score / before_slice_score
    report['after_to_before_overall_performance_ratio'] = after_overall_score / before_overall_score

    return clf, report
    
