#======================= IMPORT PACKAGES =============================

import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


#===================== 1. DATA SELECTION ==============================

#=== READ A DATASET ====

data_frame=pd.read_csv("Crop_recommendation.csv")


print("------------------------------------")
print(" 1.Data Selection ")
print("------------------------------------")
print()
print(data_frame.head(20))


#=====================  2.DATA PREPROCESSING ==========================


#=== CHECK MISSING VALUES ===

print("-------------------------------------------------------")
print("                    2.Preprocessing                  ")
print("-------------------------------------------------------")
print()
print("-------------------------------------------------------------")
print("Before Checking missing values ")
print("-------------------------------------------------------------")
print()
print(data_frame.isnull().sum())



#=== LABEL ENCODING ===

label_encoder = preprocessing.LabelEncoder() 

print("-------------------------------------------------------------")
print(" Before label encoding ")
print("------------------------------------------- ------------------")
print()
print(data_frame['label'].head(15))



data_frame['label']= label_encoder.fit_transform(data_frame['label'])

data_frame['Season']= label_encoder.fit_transform(data_frame['Season'])

data_frame['Weather Type']= label_encoder.fit_transform(data_frame['Weather Type'])


print("-------------------------------------------------------------")
print(" After label encoding ")
print("-------------------------------------------------------------")
print()
print(data_frame['label'].head(15))



#=============================== 3. DATA SPLITTING ============================

X=data_frame.drop('label',axis=1)
y=data_frame['label']


X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

print("-------------------------------------------------------------")
print(" Data Splitting ")
print("-------------------------------------------------------------")
print()
print("Total No.of data's in dataset: ", data_frame.shape[0])
print()
print("Total No.of training data's  : ", X_train.shape[0])
print()
print("Total No.of testing data's  : ", X_test.shape[0])




#============================  5. CLASSIFICATION =============================



#==========================================
# LOGISTIC REGRESSION 
#==========================================



from sklearn import linear_model


lr = linear_model.LogisticRegression()
 
lr.fit(X_train, y_train)
 
y_pred = lr.predict(X_test)

acc11=metrics.accuracy_score(y_pred,y_test)*100


cm_lr=metrics.confusion_matrix(y_pred,y_test)


print("----------------------------------------")
print("LOGISTIC REGRESSION --> LR")
print("------------------------------------")
print()
print("1. Accuracy =",acc11,'%' )
print()
print(metrics.classification_report(y_pred,y_test))

# === CONFUSION MATRIX ===

sns.heatmap(cm_lr, annot=True)
plt.title("Confusion matrix for LR")
plt.show()



#==========================================
#  XGBOOST 
#==========================================




import xgboost as xgb

xgbb=xgb.XGBClassifier()


xgbb.fit(X_train, y_train)
 
y_pred_xg = xgbb.predict(X_test)

acc_xg=metrics.accuracy_score(y_pred_xg,y_test)*100


cm_xg=metrics.confusion_matrix(y_pred_xg,y_test)


print("----------------------------------------")
print("XGBOOST CLASSIFICATION --> XGB")
print("------------------------------------")
print()
print("1. Accuracy =",acc_xg,'%' )
print()
print(metrics.classification_report(y_pred_xg,y_test))

# === CONFUSION MATRIX ===

sns.heatmap(cm_xg, annot=True)
plt.title("Confusion matrix for Xgboost")
plt.show()





#==========================================
# o HTBRID LR and RF
#==========================================


from sklearn.ensemble import VotingClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


estimator = [] 
estimator.append(('LR',  
                  LogisticRegression(solver ='lbfgs',  
                                     multi_class ='multinomial',  
                                     max_iter = 200))) 
estimator.append(('GB', GradientBoostingClassifier())) 
  
# Voting Classifier with hard voting 
vot_hard = VotingClassifier(estimators = estimator, voting ='hard') 
vot_hard.fit(X_train, y_train) 
y_pred_hyb = vot_hard.predict(X_train) 

acc_hyb = metrics.accuracy_score(y_pred_hyb,y_train) * 100

loss_hyb= 100 - acc_hyb

print("---------------------------------------------")
print("  Performance Analysis - LOGISTIC REGRESSION")
print("---------------------------------------------")

print()

print("1) Accuracy = " ,acc_hyb )
print()
print("2) Loss     = ",loss_hyb )
print()
print("3) Classification Report")
print()
print(metrics.classification_report(y_pred_hyb,y_train))
        



# ------------------ COMPARISON GRAPH 


import seaborn as sns
sns.barplot(x=['Logistic Regression','XGBOOST','Hybrid'],y=[acc11,acc_xg,acc_hyb])
plt.title("Comparion Graph")
plt.show()



import pickle
with open('model.pickle', 'wb') as f:
    pickle.dump(vot_hard, f)





