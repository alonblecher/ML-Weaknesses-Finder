import pandas as pd
import pprint
from CustomFreyaAI import CustomFreyaAI

df = pd.read_csv('./datasets/bank-additional-full.csv', delimiter=';')
target = 'y'
df[target].replace(('yes', 'no'), (1, 0), inplace=True)
cf = CustomFreyaAI(df, target)
feature_slices, largest_feature_error_rates = cf.get_slices_report({'categorical_threshold': 0.001, 'eps': 0.05, 'hdr_threshold': 0.001})
field = "emp.var.rate_housing"
clf, report = cf.apply_solution(field, "reweighting", {'weight':50})
pprint.pprint(report)
print("--------------------------------")
clf, report = cf.apply_solution(field, "Ad_Hoc", {'generator_sample_size': 200})
pprint.pprint(report)
print("--------------------------------")
clf, report = cf.apply_solution(field, "data_synthesizer", {'generator_sample_size': 500})
pprint.pprint(report)