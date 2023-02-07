import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error,accuracy_score,roc_auc_score,precision_score, recall_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('code/clean_data.csv', index_col=None)

df[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16',
'x17','x18','x19','x20','x21','x22','x23','x24','x25','x26','x27','x28','x29','x30','x31','x32','x33','x34',
'x35','x36','x37','x38','x39','x40','x41','x42','x43','x44','x45','x46','x47','x48','x49','x50']] = df[['x1',
'x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16',
'x17','x18','x19','x20','x21','x22','x23','x24','x25','x26','x27','x28','x29','x30','x31','x32','x33','x34',
'x35','x36','x37','x38','x39','x40','x41','x42','x43','x44','x45','x46','x47','x48','x49','x50']].astype(float)


df.x34.fillna(0, inplace=True)
df.x35.fillna(0, inplace=True)
df.x36.fillna(0, inplace=True)
df.x37.fillna(0, inplace=True)
df.x38.fillna(0, inplace=True)
df.x39.fillna(0, inplace=True)
df.x40.fillna(0, inplace=True)
df.x41.fillna(0, inplace=True)
df.x42.fillna(0, inplace=True)
df.x43.fillna(0, inplace=True)
df.x44.fillna(0, inplace=True)
df.x45.fillna(0, inplace=True)
df.x46.fillna(0, inplace=True)
df.x47.fillna(0, inplace=True)
df.x48.fillna(0, inplace=True)
df.x49.fillna(0, inplace=True)
df.x50.fillna(0, inplace=True)


std_scaler = StandardScaler()

df[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16',
'x17','x18','x19','x20','x21','x22','x23','x24','x25','x26','x27','x28','x29','x30','x31','x32','x33','x34',
'x35','x36','x37','x38','x39','x40','x41','x42','x43','x44','x45','x46','x47','x48','x49',
'x50']] = std_scaler.fit_transform(df[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16',
'x17','x18','x19','x20','x21','x22','x23','x24','x25','x26','x27','x28','x29','x30','x31','x32','x33','x34',
'x35','x36','x37','x38','x39','x40','x41','x42','x43','x44','x45','x46','x47','x48','x49','x50']])

le = LabelEncoder()
df['Known_Allergen']= le.fit_transform(df['Known_Allergen'])

y = df['Known_Allergen']
X = df[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16',
'x17','x18','x19','x20','x21','x22','x23','x24','x25','x26','x27','x28','x29','x30','x31','x32','x33','x34',
'x35','x36','x37','x38','x39','x40','x41','x42','x43','x44','x45','x46','x47','x48','x49','x50']]


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.05,random_state=41)

def train_models(X_train, y_train):
    
    #use Decision Tree

    tree = DecisionTreeClassifier(max_depth = 35, random_state = 0)
    tree.fit(X_train, y_train)
    y_pred_tree = tree.predict(X_test)
  
    #use the RandomForest classifier

    rf = RandomForestClassifier(n_estimators = 200, max_features = 32)
    rf.fit(X_train, y_train)
    y_pred_rf= rf.predict(X_test)

    #from sklearn.svm import SVC
    svr= SVC(kernel = 'rbf')
    svr.fit(X_train, y_train)
    y_pred_svc = svr.predict(X_test)
    
    #from sklearn.svm import SVC
    svr_p= SVC(kernel = 'linear')
    svr_p.fit(X_train, y_train)
    y_pred_svc_poly = svr_p.predict(X_test)

    # use the knn classifier
    knn = neighbors.KNeighborsClassifier(n_neighbors=3, weights= 'distance')
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)

    # Using gradient boosted machine
    gbm = GradientBoostingClassifier(n_estimators=200, min_samples_split=4, max_depth=7, learning_rate=.05)
    gbm.fit(X_train,y_train)
    y_pred_gbm = gbm.predict(X_test)
    
    # Using a logistic regression
    logit = LogisticRegression()
    logit.fit(X_train,y_train)
    y_pred_logit = logit.predict(X_test)
    AUC_logit = roc_auc_score(y_test, y_pred_logit)

  # metrics of decision tree classifier
    meanAbErr_tree= mean_absolute_error(y_test, y_pred_tree)
    meanSqErr_tree= mean_squared_error(y_test, y_pred_tree)
    rootMeanSqErr_tree= np.sqrt(mean_squared_error(y_test, y_pred_tree))
    precision_tree = precision_score(y_test,y_pred_tree)
    recall_tree = recall_score(y_test,y_pred_tree)
    AUC_tree = roc_auc_score(y_test, y_pred_tree)

  # metrics of random forest classifier
    meanAbErr_rf= mean_absolute_error(y_test, y_pred_rf)
    meanSqErr_rf= mean_squared_error(y_test, y_pred_rf)
    rootMeanSqErr_rf= np.sqrt(mean_squared_error(y_test, y_pred_rf))
    precision_rf = precision_score(y_test,y_pred_rf)
    recall_rf = recall_score(y_test,y_pred_rf)
    AUC_rf = roc_auc_score(y_test, y_pred_rf)

    # metrics of knn classifier
    meanAbErr_knn = mean_absolute_error(y_test, y_pred_knn)
    meanSqErr_knn = mean_squared_error(y_test, y_pred_knn)
    rootMeanSqErr_knn= np.sqrt(mean_squared_error(y_test, y_pred_knn))
    precision_knn = precision_score(y_test,y_pred_knn)
    recall_knn = recall_score(y_test,y_pred_knn)
    AUC_knn = roc_auc_score(y_test, y_pred_knn) 

  # metrics of svc_rbf
    meanAbErr_svc_rbf = mean_absolute_error(y_test, y_pred_svc)
    meanSqErr_svc_rbf = mean_squared_error(y_test, y_pred_svc)
    rootMeanSqErr_svc_rbf= np.sqrt(mean_squared_error(y_test, y_pred_svc))
    precision_svc_rbf = precision_score(y_test,y_pred_svc)
    recall_svc_rbf = recall_score(y_test,y_pred_svc)
    AUC_svc_rbf = roc_auc_score(y_test, y_pred_svc) 

 # metrics of svc_poly
    meanAbErr_svc_poly = mean_absolute_error(y_test, y_pred_svc_poly)
    meanSqErr_svc_poly = mean_squared_error(y_test, y_pred_svc_poly)
    rootMeanSqErr_svc_poly= np.sqrt(mean_squared_error(y_test, y_pred_svc_poly))
    precision_svc_poly = precision_score(y_test,y_pred_svc_poly)
    recall_svc_poly = recall_score(y_test,y_pred_svc_poly)
    AUC_svc_poly = roc_auc_score(y_test, y_pred_svc_poly) 

  # metrics of gbm
    meanAbErr_gbm = mean_absolute_error(y_test, y_pred_gbm)
    meanSqErr_gbm = mean_squared_error(y_test, y_pred_gbm)
    rootMeanSqErr_gbm= np.sqrt(mean_squared_error(y_test, y_pred_gbm))
    precision_gbm = precision_score(y_test,y_pred_gbm)
    recall_gbm = recall_score(y_test,y_pred_gbm)
    AUC_gbm = roc_auc_score(y_test, y_pred_gbm) 

  # metrics of logit
    meanAbErr_logit = mean_absolute_error(y_test, y_pred_logit)
    meanSqErr_logit = mean_squared_error(y_test, y_pred_logit)
    rootMeanSqErr_logit= np.sqrt(mean_squared_error(y_test, y_pred_logit))
    precision_logit = precision_score(y_test,y_pred_logit)
    recall_logit = recall_score(y_test,y_pred_logit)
    AUC_logit = roc_auc_score(y_test, y_pred_logit) 

  #print the training accurancy of each model:
    print('[1]Decision Tree Training Accuracy: ', accuracy_score(y_test,y_pred_tree))
    print('Precision Score', precision_tree)
    print('Recall Score', recall_tree)
    print('Mean Absolute Error:', meanAbErr_tree)
    print('Mean Square Error:', meanSqErr_tree)
    print('Root Mean Square Error:', rootMeanSqErr_tree)
    print('AUC:', AUC_tree)
    print('\t')
    print('[2]RandomForestClassifier Training Accuracy: ',accuracy_score(y_test,y_pred_rf))
    print('Precision Score', precision_rf)
    print('Recall Score', recall_rf)
    print('Mean Absolute Error:', meanAbErr_rf)
    print('Mean Square Error:', meanSqErr_rf)
    print('Root Mean Square Error:', rootMeanSqErr_rf)
    print('AUC:', AUC_rf)
    print('\t')    
    print('[3]SupportvectorClassifier Accuracy(rbf): ', accuracy_score(y_test,y_pred_svc))
    print('Precision Score', precision_svc_rbf)
    print('Recall Score', recall_svc_rbf)
    print('Mean Absolute Error:', meanAbErr_svc_rbf)
    print('Mean Square Error:', meanSqErr_svc_rbf)
    print('Root Mean Square Error:', rootMeanSqErr_svc_rbf)
    print('AUC:', AUC_svc_rbf)
    print('\t')
    print('[4]SupportvectorClassifier Accuracy(poly): ', accuracy_score(y_test,y_pred_svc_poly))
    print('Precision Score', precision_svc_poly)
    print('Recall Score', recall_svc_poly)
    print('Mean Absolute Error:', meanAbErr_svc_poly)
    print('Mean Square Error:', meanSqErr_svc_poly)
    print('Root Mean Square Error:', rootMeanSqErr_svc_poly)
    print('AUC:', AUC_svc_poly)
    print('\t')
    print('[5]knn Training Accuracy: ', accuracy_score(y_test,y_pred_knn))
    print('Precision Score', precision_knn)
    print('Recall Score', recall_knn)
    print('Mean Absolute Error:', meanAbErr_knn)
    print('Mean Square Error:', meanSqErr_knn)
    print('Root Mean Square Error:', rootMeanSqErr_knn)
    print('AUC:', AUC_knn)
    print('\t')
    print('[6]gbm Training Accuracy: ', accuracy_score(y_test,y_pred_gbm))
    print('Precision Score', precision_gbm)
    print('Recall Score', recall_gbm)
    print('Mean Absolute Error:', meanAbErr_gbm)
    print('Mean Square Error:', meanSqErr_gbm)
    print('Root Mean Square Error:', rootMeanSqErr_gbm)
    print('AUC:', AUC_gbm)
    print('\t')
    print('[6]logit Training Accuracy: ', accuracy_score(y_test,y_pred_logit))
    print('Precision Score', precision_logit)
    print('Recall Score', recall_logit)
    print('Mean Absolute Error:', meanAbErr_logit)
    print('Mean Square Error:', meanSqErr_logit)
    print('Root Mean Square Error:', rootMeanSqErr_logit)
    print('AUC:', AUC_logit)
    print('\t')
    
train_models(X_train,y_train)