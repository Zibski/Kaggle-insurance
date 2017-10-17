'''
Created on 12 paz 2017

@author: Zbyszek
'''
import pandas as pd
from catboost import CatBoostClassifier
import os
from tqdm import tqdm
from datetime import datetime
#import matplotlib.pyplot as plt
import data_loader

os.chdir('C:\kaggle\Porto Seguro')

columns_to_keep = ['id',
'ps_ind_03',
'ps_ind_05_cat',
'ps_ind_06_bin',
'ps_ind_15',
'ps_ind_17_bin',
'ps_reg_01',
'ps_reg_03',
'ps_car_13']

data_files = data_loader.load_all_data_files()
test_df = data_loader.load_data(data_files[1])
train_df = data_loader.load_data(data_files[2])

print("Train: ", train_df.shape)
print("Test: ", test_df.shape)

test_df = test_df[columns_to_keep]
columns_to_keep.append('target')
columns_to_keep.append('id')
train_df = train_df[columns_to_keep]

print("Train after reduction: ", train_df.shape)
print("Test after reduction: ", test_df.shape)

exclude_missing = []
exclude_other = ['id', 'target']
exclude_unique = []

train_features = []
for c in train_df.columns:
    if c not in exclude_missing \
       and c not in exclude_other and c not in exclude_unique:
        train_features.append(c)
print("We use these for training: %s" % train_features)
print(len(train_features))


cat_feature_inds = []
for i, c in enumerate(train_features):
    num_uniques = len(train_df[c].unique())
    if 'cat' in c:
        cat_feature_inds.append(i)
     
        
print("Cat features are: %s" % [train_features[ind] for ind in cat_feature_inds])

X_train = train_df[train_features]
y_train = train_df.target
X_test = test_df[train_features]
print('Test shape: {}'.format( X_test.shape))
print(X_train.shape, y_train.shape)

num_ensembles = 10
y_pred = 0.0

for i in tqdm(range(num_ensembles)):
    model = CatBoostClassifier(depth=5, iterations=600, l2_leaf_reg=6, learning_rate=0.059)
    model.fit(X_train, y_train, cat_features=cat_feature_inds)
    y_pred += model.predict_proba(X_test)[:,1]
    
    
y_pred /= num_ensembles

submission = pd.DataFrame({
    'id': test_df['id']
})
submission['target'] = y_pred

data_loader.save_data(submission, 'sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')))
print("Done! Good luck with submission :)")


#plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
#print(list(zip(X_test.columns, model.feature_importances_)))


