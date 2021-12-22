import pandas as pd
import pprint
from CustomFreyaAI import CustomFreyaAI
columns = ['Account Balance', 'Duration of Credit (month)',
       'Payment Status of Previous Credit', 'Purpose', 'Credit Amount',
       'Value Savings/Stocks', 'Length of current employment',
       'Instalment per cent', 'Sex & Marital Status', 'Guarantors',
       'Duration in Current address', 'Most valuable available asset',
       'Age (years)', 'Concurrent Credits', 'Type of apartment',
       'No of Credits at this Bank', 'Occupation', 'No of dependents',
       'Telephone', 'Foreign Worker', 'Creditability']
df = pd.read_csv('./datasets/german.data', header = None, delimiter= ' ', names = columns)
target = 'Creditability'
df[target].replace((2), (0), inplace=True)
cf = CustomFreyaAI(df, target)
feature_slices, largest_feature_error_rates = cf.get_slices_report({'categorical_threshold': 0.001, 'eps': 0.05, 'hdr_threshold': 0.001})
field = "Payment Status of Previous Credit_Instalment per cent"
clf, report = cf.apply_solution(field, "reweighting", {'weight':50})
pprint.pprint(report)
print("--------------------------------")
clf, report = cf.apply_solution(field, "Ad_Hoc", {'generator_sample_size': 200})
pprint.pprint(report)
print("--------------------------------")
clf, report = cf.apply_solution(field, "data_synthesizer", {'generator_sample_size': 500})
pprint.pprint(report)