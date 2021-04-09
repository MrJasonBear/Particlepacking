# Import Modulues
#==================================
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from matplotlib import cm
from collections import OrderedDict

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score

from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.utils import shuffle

from sklearn import preprocessing
from sklearn import utils

import scipy.interpolate as interp

# %%===============================
#           Functions
#==================================
# Split a dataset based on an attribute and an attribute value
# Sorts the Training and Testing Datasets based upon a radii size
def test_split(index, value, dataset, num):
    train, test = list(), list() 
    for loca in index:
        t=-1
        for row in dataset.iloc[:,loca]:
            t=t+1
            if row == value:
                test.append(num[t])
                
    train = list(set(num)-set(test))
    return test, train

def test_split_MF(index, value, dataset, num):
    train, test = list(), list() 
    t=-1
    for row in dataset.iloc[:,index]:
        t=t+1
        if value == num[t]:
            test.append(t)
                
    train = list(set(dataset.iloc[:,0])-set(test))
    return test, train

def test_split_wt(index, value, dataset, num):
    train, test = list(), list() 
    for loca in index:
        t=-1
        for row in dataset.iloc[:,loca]:
            t=t+1
            if row in value:
                test.append(num[t])
                
    train = list(set(num)-set(test))
    test = list(set(num)-set(train))
    return test, train

# Identifies the different unique values in a list
def searchValue(index, dataset):
    seen, values = set(), list()
    uniq = []
    for x in dataset.iloc[:,index]:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    uniq.sort()  
    values = uniq
    return values

## Split into regions of Mass Fration, Volume, Standard Deviations
#def search_MVS(dataset,Rcln,Rclp,Rcun,Rcup):
#    

# %%===============================
#           Obtains Data 
#==================================
#df = pd.read_excel('MachineLearning_13280_Reduced.xlsx')
df = pd.read_excel('Generation 4.xlsx')
df_perm = df

# Separates data 
#==================================
Run = df['Run ']
ID = df['ID ']
df = df.drop(['Run '],axis=1)               #Remove .stat file number
X = df.drop(['Packing_Fraction '],axis=1)   #Inputs

Xt = X.drop(['ID '],axis=1)                 #All features
y = df['Packing_Fraction ']                 #Packing fraction, Output
num = df['ID ']                             #Number in excel file read under


# %%
# =============================================================================
# MOST EXCEPTIONAL SETUP
# =============================================================================

df_seven = df.drop(['ID '],axis=1)

              
df_main = df
df_main.sort_values(by='Packing_Fraction ',ascending=False)

#                 Main Test Train Split
# =============================================================================
cutoff = 499        #number of exceptional values
split = 0.25        #percentage used for testing

exceptional = df_main.iloc[0:cutoff, :]
normal = df_main.iloc[cutoff+1 :, :]

df_extra1 = exceptional.sample(frac=split,replace=False)
df_extra2 = exceptional[~exceptional.isin(df_extra1)].dropna()

df_norm1 = normal.sample(frac=split,replace=False)
df_norm2 = normal[~normal.isin(df_norm1)].dropna()

df_test = pd.concat([df_extra1, df_norm1])                  #TESTING DATA
df_train_intermediate = pd.concat([df_extra2, df_norm2])

df_Training_y = df_train_intermediate.iloc[:,-1]
df_Training_X = df_train_intermediate.drop(['Packing_Fraction '],axis=1)

#                 Training Data Split
# =============================================================================
df_train_intermediate.sort_values(by='Packing_Fraction ',ascending=False)

cutoff2 = int(cutoff*(1-split))         #Number of exceptional passed into training data
excep_train = df_train_intermediate.iloc[0:cutoff2, :]  #remainder of exceptional 
norm_train = df_train_intermediate.iloc[cutoff2+1 :, :] #remainder of normal

split2 = 0.5    #splits the data evenly
df_extra_val = excep_train.sample(frac=split2,replace=False)
df_extra_train = excep_train[~excep_train.isin(df_extra_val)].dropna()

df_norm_val = norm_train.sample(frac=split2,replace=False)
df_norm_train = norm_train[~norm_train.isin(df_norm_val)].dropna()


df_validate = pd.concat([df_extra_val, df_norm_val])        #VALIDATION DATA

#==============================================================================
df_training = pd.concat([df_extra_train, df_norm_train])    #TRAINING DATA

df_train_y = df_training.iloc[:,-1]                         #Train Packing Fraction
df_train = df_training.drop(['Packing_Fraction '],axis=1)   #Train Inputs

df_validate_y = df_validate.iloc[:,-1]                         #Validate Packing Fraction
df_validate = df_validate.drop(['Packing_Fraction '],axis=1)   #Validate Inputs

df_test_y = df_test.iloc[:,-1]                         #Test Packing Fraction
df_test = df_test.drop(['Packing_Fraction '],axis=1)   #Test Inputs


# %%===============================
#        Evaluate Algorithm
#==================================
index = [0, 1, 2]                           #Index for weight averaged radii

trainset = 0
tests = 0
stored = list()
train, test = list(), list() 

#==================================
#          Predictions
#==================================
predictions = []
real_value = []
real_value2 = []
predictions_train = []
real_value_train = []

#################################################################
####################   KAAI ADDED THIS   ########################
#################################################################
def label_data(df_y, N):
    # sort the values
    df_y_sorted = df_y.sort_values()
    # manually take the top N compounds and label them as 1 for extraordinary
    df_y_sorted.iloc[-N:] = [1] * N
    # manually label all compounds below the top N as 0 for ordinary
    df_y_sorted.iloc[:-N] = [0] * (df_y.shape[0] - N)
    # resort the data so that the index matches originial df_train_y
    df_y_sorted = df_y_sorted.loc[df_y.index.values]
    return df_y_sorted
#################################################################
#################################################################
    
# %%  Training/Validation
#n = 100
#y_train = label_data(df_train_y, N=n) ###################################
#X_train = df_train
#
#y_test = label_data(df_validate_y, N=n)  ###################################
#X_test = df_validate
#
## Training
#rf = RandomForestClassifier(n_estimators=100)
#rf.fit(X_train, y_train)
#
## Validation
#prediction = rf.predict_proba(X_test)
    
# %% Added From Regressor
# =============================================================================
# MOST EXCEPTIONAL SETUP
# =============================================================================

df = df.drop(['Wt_Avg_Pt#_1_Size '],axis=1)               #Remove .stat file number
df = df.drop(['Wt_Avg_Pt#_2_Size '],axis=1)               #Remove .stat file number
df = df.drop(['Wt_Avg_Pt#_3_Size '],axis=1)               #Remove .stat file number
df = df.drop(['Wt_Avg_Pt#_1_Fraction '],axis=1)               #Remove .stat file number
df = df.drop(['Wt_Avg_Pt#_2_Fraction '],axis=1)               #Remove .stat file number
df = df.drop(['Wt_Avg_Pt#_3_Fraction '],axis=1)               #Remove .stat file number
              
df_main = df
df_main.sort_values(by='Packing_Fraction ',ascending=False)

#                 Main Test Train Split
# =============================================================================
cutoff = 499        #number of exceptional values
split = 0.5        #percentage used for testing

exceptional = df_main.iloc[0:cutoff, :]
normal = df_main.iloc[cutoff+1 :, :]

df_extra1 = exceptional.sample(frac=split,replace=False)
df_extra2 = exceptional[~exceptional.isin(df_extra1)].dropna()

df_norm1 = normal.sample(frac=split,replace=False)
df_norm2 = normal[~normal.isin(df_norm1)].dropna()

df_test = pd.concat([df_extra1, df_norm1])                  #TESTING DATA
df_test_y = df_test.iloc[:,-1]                         #Validate Packing Fraction
df_test = df_test.drop(['Packing_Fraction '],axis=1)   #Validate Inputs
df_test = df_test.drop(['ID '],axis=1)


df_train_intermediate = pd.concat([df_extra2, df_norm2])
df_train_intermediate_y = df_train_intermediate.iloc[:,-1]   #Y data
df_train_intermediate = df_train_intermediate.drop(['Packing_Fraction '],axis=1)   #Validate Inputs
df_train_intermediate = df_train_intermediate.drop(['ID '],axis=1)
# %%  Training/Validation
n = 100
y_train = label_data(df_train_intermediate_y, N=n) ###################################
X_train = df_train_intermediate

y_test = label_data(df_test_y, N=n)  ###################################
X_test = df_test

# Training
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Validation
prediction = rf.predict_proba(X_test)
# %%##########################################################
# probabilities are returned for label 0, and 1. Grab prob for label 
prob = [pred[1] for pred in prediction]  # "list comporehension" to get prob
##############################################################

xtest_prob = X_test.copy()
xtest_prob['Probisgreat'] = prob    #Prob is great controls the hue

real_value += list(df_test_y)

#xtest_prob_samp = xtest_prob.sample(100)
#import seaborn
#g = seaborn.pairplot(xtest_prob_samp, hue="Probisgreat")


rv = pd.DataFrame({'rv':real_value.copy()})
prb = pd.DataFrame({'prb':prob.copy()})

f_matrix = pd.concat([rv, prb], axis=1, join='inner')

tp = int()
tn = int()
fn = int()
fp = int()

prob_lim = 0.333
extra_lim = 0.785

for index, row in f_matrix.iterrows():
    if row['rv'] > extra_lim:
        if row['prb'] > prob_lim:
            tp += 1
        else:
            fn += 1
    else:
        if row['prb'] > prob_lim:
            fp += 1
        else:
            tn += 1

#Data that has a probability of being extraordinary
#=======================================================
xtest_prob_reduced = xtest_prob[xtest_prob["Probisgreat"] > 0.3]

#interp.griddata(xtest_prob)

# ===============================
##             Plotting    
##==================================
fig = plt.figure(1, figsize=(12, 12))
ax = fig.add_axes([0,0,1,1])

## updated it to plot against the probility of label 1
data = plt.plot(real_value, prob, 'ro', markersize=12, alpha=0.3)
ax.tick_params(direction='out', labelsize = 25, length=10, width=3, grid_color ='k')

#plt.title('Random Forrest Classifer',
#          fontsize = 25, weight = 'bold')
plt.xlabel('Actual Value', fontsize = 25, weight = 'bold')
plt.ylabel('Probability of Being Extraordinary', fontsize =25, weight = 'bold')
plt.grid(True)

#==============Legend Details==================
FN = 'False Negative: '
FN += str(fn)
rect_fn = plt.Rectangle((extra_lim,prob_lim), 0.11, (-prob_lim),color='b',
                        alpha = 0.3,ec='k',label=str(FN))
ax.add_patch(rect_fn)

FP = 'False Positive: '
FP += str(fp)
rect_fp = plt.Rectangle((0.5,prob_lim), (extra_lim-0.5), (1-prob_lim),color='r',
                        alpha = 0.3,ec='k',label=str(FP))
ax.add_patch(rect_fp)

TP = 'True Positive: '
TP += str(tp)
rect_tp = plt.Rectangle((extra_lim,prob_lim), 0.11, (1-prob_lim),color='g',
                        alpha = 0.3,ec='k',label=str(TP))
ax.add_patch(rect_tp)

TN = 'True Negative: '
TN += str(tn)
rect_tn = plt.Rectangle((0.5,0), (extra_lim-0.5), (prob_lim),color='w',
                        alpha = 1,ec='k',label=str(TN))
ax.add_patch(rect_tn)

accuracy = (tp+tn)/(tp+fp+tn+fn)
acc = float('%.4g' %accuracy)
Acc = 'Accuracy: '
Acc += str(acc)
plt.plot([], [], ' ', label = str(Acc))

precision = tp/(tp+fp)
recall = tp/(fn+tp)

F1 = 2 * (precision*recall)/(precision+recall)
F1 = float('%.4g' %F1)
F1score = 'F1 Score: '
F1score += str(F1)

plt.plot([], [], ' ', label = str(F1score))
plt.legend(fontsize = 25)
#plt.legend(fontsize = 'xx-large')

plt.xlim(0.5, 0.85)
plt.ylim(0, 1)
plt.show()

# %% Predictions
#df2 = pd.read_excel('Files for Predictions.xlsx')
df2 = pd.read_excel('Partial 2-model.xlsx')
#run = df2['Run '] 
#run = run.to_frame()
#df2 = df2.drop(['Run '],axis=1)
#df2 = df2.drop(['ID '],axis=1)

y_predicted = rf.predict(df2)
y_predicted = pd.DataFrame({'Packing Fraction':y_predicted})

# Write predicted cases
ans = pd.concat([df2,y_predicted], axis=1, sort=False)
#ans.to_excel("Generation-4-Classifier-Partial 2-model.xlsx")
