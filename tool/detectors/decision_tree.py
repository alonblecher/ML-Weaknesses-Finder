import numpy as np
import pandas as pd
import graphviz
from sklearn import tree


def get_lineage(tree, feature_names):
     left = tree.tree_.children_left
     right     = tree.tree_.children_right
     threshold = tree.tree_.threshold
     features  = [feature_names[i] if i != -2 else -5 for i in tree.tree_.feature]

     # get ids of child nodes
     idx = np.argwhere(left == -1)[:,0]     

     def recurse(left, right, child, lineage=None):          
          if lineage is None:
               lineage = [child]
          if child in left:
               parent = np.where(left == child)[0].item()
               split = 'l'
          else:
               parent = np.where(right == child)[0].item()
               split = 'r'

          lineage.append((parent, split, threshold[parent], features[parent]))

          if parent == 0:
               lineage.reverse()
               return lineage
          else:
               return recurse(left, right, parent, lineage)
    
     childs = {}

     for child in idx:
          child_rules = []
          for node in recurse(left, right, child):
               child_rules.append(node)
          childs[child] = child_rules
     return childs

def get_leaves_props(leaves, tree_clf):
  leaves_range_dict = {}
  for key,value in leaves.items():
    range_min_max_dict = {}
    for range_tuple in value:
      if type(range_tuple) == tuple:
        if range_tuple[3] not in range_min_max_dict:
          range_min_max_dict[range_tuple[3]] = {}
        if (range_tuple[1] == 'r'):
          if 'min' not in range_min_max_dict[range_tuple[3]]:
            range_min_max_dict[range_tuple[3]]['min'] = range_tuple[2]
          else:
            if range_tuple[2] < range_min_max_dict[range_tuple[3]]['min']:
              range_min_max_dict[range_tuple[3]]['min'] = range_tuple[2]
        elif (range_tuple[1] == 'l'):
          if 'max' not in range_min_max_dict[range_tuple[3]]:
            range_min_max_dict[range_tuple[3]]['max'] = range_tuple[2]
          else:
            if range_tuple[2] > range_min_max_dict[range_tuple[3]]['max']:
              range_min_max_dict[range_tuple[3]]['max'] = range_tuple[2]
    leaves_range_dict[key] = {}
    leaves_range_dict[key]['range'] = range_min_max_dict
    leaves_range_dict[key]['relative_error_rate'] = tree_clf.tree_.value[key][0][0] / tree_clf.tree_.value[0][0][0]
    leaves_range_dict[key]['error_rate'] = tree_clf.tree_.value[key][0][0] / (tree_clf.tree_.value[key][0][0] + tree_clf.tree_.value[key][0][1])
  return leaves_range_dict

def get_stat_important_leaves(leaves, tree_clf):
  return dict(filter(lambda elem: elem[1]['relative_error_rate'] * tree_clf.tree_.value[0][0][1] > max(2, 0.05 * tree_clf.tree_.value[0][0][1]), leaves.items()))

def get_decision_tree_slices(X, Y, cols):
  tree_clf = tree.DecisionTreeClassifier()
  tree_clf = tree_clf.fit(X[cols], Y)
  dot_data = tree.export_graphviz(tree_clf, out_file=None, 
                      filled=True, rounded=True,  
                      special_characters=True)  
  graph = graphviz.Source(dot_data)  
  leaves = get_lineage(tree_clf, X[cols].columns)
  slices = get_stat_important_leaves(get_leaves_props(leaves, tree_clf), tree_clf)
  slices_list = []
  for key in slices.keys():
    slices_list.append(slices[key])
  return slices_list, graph

def get_slices_by_range(df, range):
  vec = np.ones(df.shape[0], dtype=bool)
  for key in range.keys():
    vec = vec & df[key].between(range[key].get('min', float('-inf')), range[key].get('max', float('inf')))
  return df[vec]

def tree_indexes_by_slices(df, slices):
  if len(slices) == 0:
    return []
  df_slices = []
  for slice_ in slices:
    for inner_slice in slice_:
      df_range = get_slices_by_range(df, inner_slice['range'])
      df_slices.append(df_range)
  return pd.concat(df_slices)