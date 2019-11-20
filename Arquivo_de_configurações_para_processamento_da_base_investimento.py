dados = 'C:/Users/Rodrigo/Desktop/TCC/bank_customer_survey/bank_customer_survey.csv'
delimitador = ','
target = 'y'
separador_decimal = '.'
codigo_out = 'C:/Users/Rodrigo/Desktop/TCC/bank_customer_survey/ML_bank_customer_survey.py'

score = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']

Modelo; parametros
DecisionTreeClassifier;{'max_depth':[2,4,8,12,16], 'min_samples_split':[0.1, 1.0, 10], 'min_samples_leaf':[0.1, 0.5, 5], 'max_features':[int(X_train.shape[1]*0.25), int(X_train.shape[1]*0.50), int(X_train.shape[1]*0.75), int(X_train.shape[1])]}
XGBClassifier;{'nthread':[4],'objective':['binary:logistic'], 'learning_rate': [0.05], 'max_depth': [8,9,10], 'min_child_weight': [12,20,30], 'silent': [1],'subsample': [0.5,0.6,0.7], 'colsample_bytree': [0.6,0.7,0.8], 'n_estimators': [5,10,20] }
RandomForestClassifier;{'n_estimators':[10,50,100], 'max_depth':[2,8,16], 'min_samples_split':[0.1, 1.0, 10], 'min_samples_leaf':[0.1, 0.5, 5], 'bootstrap':[True, False], 'max_features':[int(X_train.shape[1]*0.25), int(X_train.shape[1]*0.50), int(X_train.shape[1]*0.75), int(X_train.shape[1])]}
GradientBoostingClassifier;{'learning_rate':[1, 0.25, 0.01], 'n_estimators':[4, 30], 'max_depth':[2,8,16], 'min_samples_split':[0.1, 1.0, 10], 'min_samples_leaf':[0.1, 0.5, 5], 'max_features':[int(X_train.shape[1]*0.25), int(X_train.shape[1]*0.50), int(X_train.shape[1]*0.75), int(X_train.shape[1])] }
KNeighborsClassifier;{'n_neighbors':[2,8,16,32], 'p':[1, 3, 5] }
LogisticRegression;{'C':[0.001,0.01,0.1,1,10,100] }
