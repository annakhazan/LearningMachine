
# coding: utf-8

# # A Study of Feature Importance in the Forest Cover Type Prediction Dataset
# 
# 
# Data source: https://www.kaggle.com/c/forest-cover-type-prediction

# In[1]:

import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import cross_validation
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
from IPython.core.display import display, HTML
from datetime import datetime
from sklearn.model_selection import GridSearchCV
get_ipython().magic('matplotlib inline')
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')
display(HTML("<style>.container { width:100% !important; }</style>"))


# ## Load Data

# In[2]:

forest = pd.read_csv("data/train.csv") 
forest = forest.iloc[:,1:]


# In[3]:

original_cols = list(forest.columns)
original_cols.remove('Cover_Type')


# # Feature Engineering

# In[4]:

def labelSoilType(row):
    """
    Label soil types
    """
    for i in range(len(row)):
        if row[i] == 1:
            return 'Soil_Type'+str(i)
        
def azimuth_to_abs(x):
    """
    Only care about the absolute angle from 0 w/o respect to direction
    """
    if x>180:
        return 360-x
    else:
        return x


# In[5]:

# Create Soil Type Buckets
soil_types = pd.read_csv('soil_types.csv').set_index('Soil Type')
forest['Soil Type'] = forest[['Soil_Type1', 'Soil_Type2', 'Soil_Type3',
       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7',
       'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11',
       'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15',
       'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19',
       'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23',
       'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27',
       'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31',
       'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35',
       'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39',
       'Soil_Type40']].apply(lambda row: labelSoilType(row), axis=1)
forest = pd.merge(forest, soil_types, how='left', left_on='Soil Type', right_index=True)
del forest['Soil Type'] # Delete string column

# Create feature to that transforms azimuth to its absolute value
forest['Aspect2'] = forest.Aspect.map(azimuth_to_abs)
forest['Aspect2'].astype(int)

# Create feature that determines if the patch is above sea level
forest['Above_Sealevel'] = (forest.Vertical_Distance_To_Hydrology>0).astype(int)

# Bin the Elevation Feature: check the feature exploration notebook for motivation
bins = [0, 2600, 3100, 8000]
group_names = [1, 2, 3]
forest['Elevation_Bucket'] = pd.cut(forest['Elevation'], bins, labels=group_names)
forest['Elevation_0_2600'] = np.where(forest['Elevation_Bucket']== 1, 1, 0)
forest['Elevation_2600_3100'] = np.where(forest['Elevation_Bucket']== 2, 1, 0)
forest['Elevation_3100_8000'] = np.where(forest['Elevation_Bucket']== 3, 1, 0)
forest['Elevation_0_2600'].astype(int)
forest['Elevation_2600_3100'].astype(int)
forest['Elevation_3100_8000'].astype(int)
del forest['Elevation_Bucket']

# Create a feature for no hillshade at 3pm
forest['3PM_0_Hillshade'] = (forest.Hillshade_3pm == 0).astype(int)

#Direct distance to hydrology
forest['Direct_Distance_To_Hydrology'] = np.sqrt((forest.Vertical_Distance_To_Hydrology**2) +     (forest.Horizontal_Distance_To_Hydrology**2)).astype(float).round(2)


soil_types= ['Soil_Type1', 'Soil_Type2', 'Soil_Type3',
       'Soil_Type4', 'Soil_Type5', 'Soil_Type6',
       'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11',
       'Soil_Type12', 'Soil_Type13', 'Soil_Type14',
       'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19',
       'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23',
       'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27',
       'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31',
       'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35',
       'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39',
       'Soil_Type40', 'Cover_Type']

column_list = forest.columns.tolist()
column_list = [c for c in column_list if c[:9] != 'Soil_Type']
column_list.insert(10, 'Direct_Distance_To_Hydrology')
column_list.insert(11, 'Elevation_0_2600')
column_list.insert(12, 'Elevation_2600_3100')
column_list.insert(13, 'Elevation_3100_8000')
column_list.insert(14, 'Aspect2')
column_list.insert(15, 'Above_Sealevel')
column_list.insert(16, '3PM_0_Hillshade')
column_list.extend(soil_types)
columns = []
for col in column_list:
    if col not in columns:
        if col != 'Cover_Type':
            columns.append(col)
columns.append('Cover_Type')
        

forest = forest[columns]
forest.fillna(0,inplace=True) # Replace nans with 0 for our soil type bins
forest.shape


# ## Remove Base Features with no Modeling Value

# In[6]:

to_remove = [] # features to drop
for c in forest.columns.tolist():
    if forest[c].std() == 0:
        to_remove.append(c)
forest = forest.drop(to_remove, 1)
print("Dropped the following columns: \n")
for r in to_remove:
    print (r)


# In[7]:

original_cols_with_soil = list(forest.columns)
original_cols_with_soil.remove('Cover_Type')


# ## Add Feature Interactions

# In[8]:

for i in range(forest.shape[1]-1):
    for j in range(54):
        if i != j:
            forest[forest.columns.tolist()[i]+"_"+forest.columns.tolist()[j]] = forest[forest.columns.tolist()[i]]*forest[forest.columns.tolist()[j]]


# ## Remove Columns That Have No Value

# In[9]:

to_remove = [] # features to drop
for c in forest.columns.tolist():
    if forest[c].std() == 0:
        to_remove.append(c)
forest = forest.drop(to_remove, 1)
print("Dropped the following columns: \n")
for r in to_remove:
    print (r)
all_interacted_cols = list(forest.columns)


# ### Transform the continuous features
# ###### We will try Normalization, Standardized Scaling, and MinMax Scaling
# ###### Note: there is no need to impute any data points as this is a pretty clean data set

# In[10]:

chunk_size = 0.1 #Validation chunk size
seed = 0 # Use the same random seed to ensure consistent validation chunk usage

X_all = [] # all features
X_all_add = [] # Additionally we will make a list of subsets
rem = [] # columns to be dropped
i_rem = [] # indexes of columns to be dropped
trans_list = [] # Transformations
comb = [] # combinations
comb.append("All+1.0")

ratio_list = [1.0,0.75,0.50,0.25] #Select top 100%, 75%, 50%, 25% of features
features = [] # feature selection models
model_features = [] # names of feature selection models

#Reorder the data to have continuous variables come first
continuous = []
categorical = []
final_columns = []
for col in forest.columns.tolist():
    if col in to_remove:
        pass
    elif col == 'Cover_Type':
        pass
    elif forest[col].nunique() > 4:
        continuous.append(col)
    else:
        categorical.append(col)
final_columns.extend(continuous)
final_columns.extend(categorical)
final_columns.append('Cover_Type')
forest = forest[final_columns]
num_row, num_cols = forest.shape
cols = forest.columns
size = len(continuous) # Number of continuous columns

i_cols = []
for i in range(0, num_cols-1):
    i_cols.append(i)

#Create the data arrays for model building
val_array = forest.values
X = val_array[:,0:(num_cols-1)]
y = val_array[:,(num_cols-1)]
X_train, X_val, y_train, y_val = cross_validation.train_test_split(X, y, test_size=chunk_size, random_state=seed)
X_all.append(['Orig','All', X_train,X_val,1.0,cols[:num_cols-1],rem,i_cols,i_rem])


# In[11]:

# Standardize the data

X_temp = StandardScaler().fit_transform(X_train[:,0:size])
X_val_temp = StandardScaler().fit_transform(X_val[:,0:size])

# Recombine data
X_con = np.concatenate((X_temp,X_train[:,size:]),axis=1)
X_val_con = np.concatenate((X_val_temp,X_val[:,size:]),axis=1)

X_all.append(['StdSca','All', X_con,X_val_con,1.0,cols,rem,i_cols,i_rem])


# In[12]:

# MinMax Scale the data

X_temp = MinMaxScaler().fit_transform(X_train[:,0:size])
X_val_temp = MinMaxScaler().fit_transform(X_val[:,0:size])

# Recombine data
X_con = np.concatenate((X_temp,X_train[:,size:]),axis=1)
X_val_con = np.concatenate((X_val_temp,X_val[:,size:]),axis=1)

X_all.append(['MinMax', 'All', X_con,X_val_con,1.0,cols,rem,i_cols,i_rem])


# In[13]:

#Normalize the data

X_temp = Normalizer().fit_transform(X_train[:,0:size])
X_val_temp = Normalizer().fit_transform(X_val[:,0:size])

# Recombine data
X_con = np.concatenate((X_temp,X_train[:,size:]),axis=1)
X_val_con = np.concatenate((X_val_temp,X_val[:,size:]),axis=1)

X_all.append(['Norm', 'All', X_con,X_val_con,1.0,cols,rem,i_cols,i_rem])


# In[14]:

# Add transformation to the list
for trans,name,X,X_val,v,cols_list,rem_list,i_cols_list,i_rem_list in X_all:
    trans_list.append(trans)


# ### Create classifiers
# - Logistic Regression
# - SVM

# In[15]:

# Add Logistic Regression
n = 'Logistic Regression'
model_features.append(n)
for val in ratio_list:
    comb.append('%s+%s'%(n, val))
    features.append([n, val, LogisticRegression(random_state=seed),
        {
            'penalty':('l1', 'l2'),
            'dual':(True, False),
            'C':(1e-3, 1e-2,1e-1,1e0,1e1,1e2,1e3),
            'fit_intercept':(True, False),
            'intercept_scaling':(1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3),
            'max_iter':('newton-cg', 'lbfgs', 'liblinear', 'sag'),
            'tol':(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
            'multi_class':('ovr', 'multinomial')
        }])
    
# Add SVM
n = 'SVM'
model_features.append(n)
for val in ratio_list:
    comb.append('%s+%s'%(n, val))
    features.append([n, val, LinearSVC(random_state=seed),
        {
            'C':(1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3),
            'kernel':('linear', 'poly', 'rbf', 'sigmoid', 'precomputed'),
            'degree':(1,2,3,4),
            'gamma':('auto',1e-3, 1e-2,1e-1,1e0,1e1,1e2,1e3),
            'coef0':(1e-3, 1e-2,1e-1,1e0,1e1,1e2,1e3),
            'probability':(True,False),
            'shrinking':(True,False),
            'tol':(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
            'decision_function_shape':('ovo', 'ovr', None)
        }])
    


# # Top 100 features

# In[16]:

top100 = list(pd.read_csv('top_100.csv', header=None, names=['Feature', 'Importance'])['Feature'].values)
column_lists = {
    'original':original_cols,
    'original with soil':original_cols_with_soil,
    'top 100':top100#,
    #'all':all_interacted_cols
}


# # Grid Search

# In[17]:

#Run grid search over the different data transformations
def gridSearch(model, params, X, y):
    g = GridSearchCV(model, params, error_score=-999, verbose=1)
    g.fit(X, y)
    return g.best_estimator_, g.best_score_, g.best_params_, g.cv_results_


# # Confusion Matrix Scoring

# In[18]:

def confusion_matrix_scoring(predicted_classes, true_classes):
    conf_matrix=np.array([
        [0,5,1,7,3,2,10],
        [30,0,20,1,6,10,1],
        [3,4,0,6,2,1,8],
        [40,5,30,0,10,20,1],
        [12,2,8,3,0,1,4],
        [6,2,3,4,1,0,6],
        [50,3,40,1,20,30,0]
    ])
    data_conf=np.array([
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0]
    ])
    for i in range(len(predicted_classes)):
        data_conf[int(predicted_classes[i])-1][int(true_classes[i])-1]+=1
    score=0
    for i in range(7):
        for j in range(7):
            score+=(data_conf[i][j]*conf_matrix[i][j])
    return score


# # Run models on selected features

# In[19]:

# Determine feature importance for each model and transformation combination
with open('model_testing.txt', 'w+') as file:
    output = []
    for trans, s, X, X_val, d, cols, rem, i_cols, i_rem in X_all:
        for name,v,model,params in features:
            for c in column_lists:
                print (name)
                file.write('name : ' + str(name) + '\n')
                print (v)
                file.write('values : ' + str(v) + '\n')
                print (model)
                file.write('model : ' + str(model) + '\n')
                print (c)
                file.write('c : ' + str(cols) + '\n')

                if c == 'top 100' or v == 1.0:
                    selected_features = column_lists[c][:int(len(top100)*v)]
                    print (len(selected_features))
                    file.write('selected features : ' + str(selected_features) + '\n')

                    cols_list = [] # List of names of columns selected
                    i_cols_list = [] # Indexes of columns selected
                    rank_list =[] # Ranking of all the columns
                    rem_list = [] # List of columns not selected
                    i_rem_list = [] # Indexes of columns not selected

                    for field in cols:
                        if field in selected_features:
                            cols_list.append(field)
                            i_cols_list.append(list(cols).index(field))
                        else:
                            rem_list.append(field)
                            i_rem_list.append(list(cols).index(field))

                    #Limit training and validation dataset to just relevant columns
                    X_new = np.delete(X, i_rem_list, axis=1)
                    X_val_new = np.delete(X_val, i_rem_list, axis=1)

                    #Fit the model on selected dataset
                    model.fit(X_new, y_train)

                    #Calculate model score against true class for each sample
                    print (model.score(X_val_new, y_val))
                    file.write('model score : ' + str(model.score(X_val_new, y_val)) + '\n')
                    #Grid search
                    file.write('Grid Search Results -- \n')
                    best_estimator, best_score, best_params, cv_results = gridSearch(model, params, X_new, y_train)
                    print (best_estimator)
                    file.write('best estimator : ' + str(best_estimator) + '\n')
                    print (best_score)
                    file.write('best score : ' + str(best_score) + '\n')
                    print (best_params)
                    file.write('best params : ' + str(best_params) + '\n')
                    print (cv_results)
                    file.write('best cv results : ' + str(cv_results) + '\n')

                    print (confusion_matrix_scoring(model.predict(X_val_new), y_val))
                    file.write('conf matrix score : ' + str(confusion_matrix_scoring(model.predict(X_val_new), y_val)) + '\n')
                    file.write('\n')
                    file.write('-----------------\n')
                    file.write('\n')


                    # Append model name, array, columns selected and columns to be removed to the additional list        
                    X_all_add.append([trans,name,X_new,X_val_new,v,cols_list,rem_list,i_cols_list,i_rem_list]) 


# In[ ]:




# In[ ]:




# In[ ]:



