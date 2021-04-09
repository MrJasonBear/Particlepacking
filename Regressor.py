# Import Modulues
#==================================
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from collections import OrderedDict

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score

from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm

from matplotlib.gridspec import GridSpec


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

# %%===============================
#           Obtains Data 
#==================================
#df = pd.read_excel('MachineLearning_13280.xlsx')
#df = pd.read_excel('MachineLearning_13280_Modified.xlsx')
df = pd.read_excel('Generation 0.xlsx')
df_perm = df

PF_act = df_perm.iloc[:,-1]

# Separates data 
#==================================
Run = df['Run ']
ID = df['ID ']
df = df.drop(['Run '],axis=1)               #Remove .stat file number
X = df.drop(['Packing_Fraction '],axis=1)   #Inputs

Xt = X.drop(['ID '],axis=1)                 #All features
Xdist = Xt.iloc[:,:9]                       #No Weight Average
y = df['Packing_Fraction ']                 #Packing fraction, Output
num = df['ID ']                             #Number in excel file read under
mf1 = df.iloc[:,:-1]                        #Mass fraction, particle #1
X_wt = X.iloc[:, 10:]                       #Weigthed particle data


X_rc = X.iloc[:,20:21]
X_mvs = pd.concat([ID,X_rc], axis=1)

#==================================
#        Identifies Data
#==================================
psize = searchValue(1, df)                  #particle sizes, P1
mfsize = searchValue(7, df)                 #mass fraction, MF1
wt1size = searchValue(10, df)               #Weighted radii, Wr1
wt2size = searchValue(11, df)               #Weighted radii, Wr2
wt3size = searchValue(12, df)               #Weighted radii, Wr3

wtsize = wt1size + wt2size + wt3size
wtsize = sorted(wtsize)
kf = 10                                     #Split Size

splits = np.array_split(wtsize,kf)          #Split Array

# %%
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
split = 0.25        #percentage used for testing

exceptional = df_main.iloc[0:cutoff, :]
normal = df_main.iloc[cutoff+1 :, :]

df_extra1 = exceptional.sample(frac=split,replace=False)
df_extra2 = exceptional[~exceptional.isin(df_extra1)].dropna()

df_norm1 = normal.sample(frac=split,replace=False)
df_norm2 = normal[~normal.isin(df_norm1)].dropna()

df_test = pd.concat([df_extra1, df_norm1])                  #TESTING DATA
df_train_intermediate = pd.concat([df_extra2, df_norm2])

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


df_validate_y = df_validate.iloc[:,-1]                         #Validate Packing Fraction
df_validate = df_validate.drop(['Packing_Fraction '],axis=1)   #Validate Inputs
df_validate = df_validate.drop(['ID '],axis=1)

df_test_y = df_test.iloc[:,-1]                         #Validate Packing Fraction
df_test = df_test.drop(['Packing_Fraction '],axis=1)   #Validate Inputs
df_test = df_test.drop(['ID '],axis=1)

# %%===============================
#        Evaluate Algorithm
#==================================
index = [0, 1, 2]                           #Index for weight averaged radii
Rclp = [0, 10^2, 10^3, 10^4, 10^5]
Rcup = [10^2, 10^3, 10^4, 10^5, 10^10]
Rcln = [0, 10^-2, 10^-3, 10^-4, 10^-5]
Rcun = [10^-2, 10^-3, 10^-4, 10^-5, 10^-10]

msplit = list(range(kf))
trainset = 0
tests = 0
stored = list()
train, test = list(), list() 

#==================================
#          Manual K-fold
#==================================
predictions = []
real_value = []
predictions_train = []
real_value_train = []

for dex in msplit:
    
    value = splits[dex]
    stored = stored + test
    train, test = list(), list() 

    test, train = test_split_wt(index,value,X_wt,num)

    test = [x for x in test if x not in stored]
    train = list(set(num) - set(test))
    
    if len(test)==0:                #No data exists to test
        break

    trainset = trainset + (len(test)/len(train)*100)
    tests = tests + len(test)

    tes=np.float32(np.asarray(test))
    tra=np.float32(np.asarray(train))

#CHANGE ARRRAY VALUE FOR VARIATION
    X_train, X_test = Xdist.loc[tra,:], Xdist.loc[tes,:]
#    X_train, X_test = X_wt.loc[tra,:], X_wt.loc[tes,:]
#    X_train, X_test = Xt.loc[tra,:], Xt.loc[tes,:]
    y_train, y_test = y.loc[tra], y.loc[tes]
    X_train, X_test = X_train.iloc[:, 0:], X_test.iloc[:, 0:]

    X_train=np.float32(np.asarray(X_train))
    y_train=np.float32(np.asarray(y_train))

# Training
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)

# Validation
    prediction = rf.predict(X_test)
    predictions += list(prediction)
    real_value += list(y_test)

    prediction_train = rf.predict(X_train)
    predictions_train += list(prediction_train)
    real_value_train += list(y_train)
    
# %%===============================
#df2 = pd.read_excel('Files for Predictions.xlsx')
#run = df2['Run '] 
#run = run.to_frame()
#df2 = df2.drop(['Run '],axis=1)
#
#y_predicted = rf.predict(df2)
#y_predicted = pd.DataFrame({'Packing Fraction':y_predicted})
#
## Write predicted cases
#ans = pd.concat([run,df2,y_predicted], axis=1, sort=False)
#ans.to_excel("13280Predict-Regressor.xlsx")


# Predictions
#df2 = pd.read_excel('Files for Predictions.xlsx')
df2 = pd.read_excel('Partial 2-model.xlsx')

y_predicted = rf.predict(df2)
y_predicted = pd.DataFrame({'Packing Fraction':y_predicted})

# Write predicted cases
ans = pd.concat([df2,y_predicted], axis=1, sort=False)
ans.to_excel("Generation-4-Regressor-Partial 2-model.xlsx")


# %%===============================
#             Plotting    
#==================================
ts=trainset/kf

Rtest=r2_score(real_value,predictions)
Rtrain=r2_score(real_value_train,predictions_train)
fig = plt.figure(1, figsize=(12, 12))

#=============================================================================
textsize = 25

gs = GridSpec(5, 5)
ax_main = plt.subplot(gs[1:5, 0:4])
ax_xDist = plt.subplot(gs[0, 0:4])
ax_yDist = plt.subplot(gs[1:5, 4:5])


line, = ax_main.plot([0.45, .85], [0.45, .85], '--', color='#8da0cb', linewidth = 3)
data, = ax_main.plot(real_value, predictions, '.',color='#66c2a5', markersize=12)#, alpha=0.3)
data_train, = ax_main.plot(real_value_train, predictions_train, '.',color='#fc8d62', markersize=12)#, alpha=0.1)
#ax_main.set(xlabel="Calculated Value", ylabel="Prediction")
ax_main.set_ylabel('Prediction', fontsize=textsize, fontweight='bold', labelpad = 25) 
ax_main.set_xlabel('Calculated Value', fontsize=textsize, fontweight='bold', labelpad = 25) 
#ax_main.grid(b=None)

te='Test  - R2=''{:.3f}'.format(Rtest)
tr='Train - R2=''{:.3f}'.format(Rtrain)

ax_main.legend([line, data, data_train],['Ideal Performance',te,tr],
           shadow=True, fancybox=True, fontsize = textsize)
ax_main.tick_params(direction='out', labelsize = textsize, length=10, width=1, grid_color ='k')
ax_main.grid(False)

ax_main.set_yticks([.5,.6,.7,.8])
ax_main.set_yticks([.5,.6,.7,.8])

ax_main.set_xlim(0.475,0.85)
ax_main.set_ylim(0.475,0.85)

ax_xDist.set_xlim(0.475,0.85)
ax_yDist.set_ylim(0.475,0.85)



ax_xDist.hist(real_value,bins=100,align='mid',color='#8da0cb')
ax_xDist.set_yticks([])
ax_xDist.set_xticks([])
ax_xCumDist = ax_xDist.twinx()
ax_xCumDist.hist(real_value,bins=100,cumulative=True,histtype='step',normed=True,color='r',align='mid')
ax_xCumDist.set_yticks([])

ax_yDist.hist(predictions,bins=100,orientation='horizontal',align='mid',color='#8da0cb')
ax_yDist.set_xticks([])
ax_yDist.set_yticks([])
ax_yCumDist = ax_yDist.twiny()
ax_yCumDist.hist(predictions,bins=100,cumulative=True,histtype='step',normed=True,color='r',align='mid',orientation='horizontal')
ax_yCumDist.set_xticks([])

# %%===============================
#             Importances
#==================================

#importances = list(rf.feature_importances_)
#
## list of x locations for plotting
#x_values = list(range(len(importances)))
#
## Make a bar chart
#fig = plt.figure(1, figsize=(12, 12))
#
##CHANGE ARRRAY VALUE FOR VARIATION
#feature_list= list(Xdist.columns.values)
#
## Tick labels for x axis
#ax = fig.add_axes([0,0,1,1])
#ax.tick_params(direction='out', labelsize = 25, length=10, width=3, grid_color ='k')
#plt.xticks(x_values, feature_list, rotation='vertical')
#plt.bar(x_values, importances, orientation = 'vertical')
#
## Axis labels and title
#plt.ylabel('Importance', fontsize =24, weight = 'bold')
#plt.xlabel('Variable', fontsize =24, weight = 'bold')
#plt.title('Variable Importances', fontsize =24, weight = 'bold');
#
#
#plt.show()
