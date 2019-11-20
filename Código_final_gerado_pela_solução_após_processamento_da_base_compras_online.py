# Importando Modulos e Pacotes
import numpy as np
import pandas as pd
from tabulate import tabulate
import sklearn.model_selection as model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import datetime

# Atribuindo parametros de entrada
target = 'Revenue'

# Importando os dados de entrada
df = pd.read_csv('C:/Users/rcosin/Desktop/FIA/TCC/online_shoppers_intention/online_shoppers_intention.csv', sep = ',', decimal = '.')

# Analise de variaveis e identificacao de tratamento:
#+-------------------------+---------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+
#| Coluna                  | Tipo    | Desvio Padrao   | Media   | Coef de Variacao   | Mediana   | Coef de Centralidade   |   Perc_Distintos |   Valores_Distintos |   Perc_Null | Acao              | Desc Acao   |
#+=========================+=========+=================+=========+====================+===========+========================+==================+=====================+=============+===================+=============+
#| Administrative          | float64 | 3.32            | 2.32    | 143.1              | 1.0       | 43.1                   |             0.22 |                  27 |        0.11 | dummy_com_null    | -           |
#+-------------------------+---------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+
#| Administrative_Duration | float64 | 176.86          | 80.91   | 218.59             | 8.0       | 9.89                   |            27.06 |                3336 |        0.11 | continua_com_null | -           |
#+-------------------------+---------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+
#| Informational           | float64 | 1.27            | 0.5     | 254.0              | 0.0       | 0.0                    |             0.14 |                  17 |        0.11 | dummy_com_null    | -           |
#+-------------------------+---------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+
#| Informational_Duration  | float64 | 140.83          | 34.51   | 408.08             | 0.0       | 0.0                    |            10.21 |                1259 |        0.11 | continua_com_null | -           |
#+-------------------------+---------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+
#| ProductRelated          | float64 | 44.49           | 31.76   | 140.08             | 18.0      | 56.68                  |             2.52 |                 311 |        0.11 | continua_com_null | -           |
#+-------------------------+---------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+
#| ProductRelated_Duration | float64 | 1914.37         | 1196.04 | 160.06             | 599.77    | 50.15                  |            77.47 |                9552 |        0.11 | continua_com_null | -           |
#+-------------------------+---------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+
#| BounceRates             | float64 | 0.05            | 0.02    | 250.0              | 0.0       | 0.0                    |            15.18 |                1872 |        0.11 | continua_com_null | -           |
#+-------------------------+---------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+
#| ExitRates               | float64 | 0.05            | 0.04    | 125.0              | 0.03      | 75.0                   |            38.74 |                4777 |        0.11 | continua_com_null | -           |
#+-------------------------+---------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+
#| PageValues              | float64 | 18.57           | 5.89    | 315.28             | 0.0       | 0.0                    |            21.93 |                2704 |        0    | continua_sem_null | -           |
#+-------------------------+---------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+
#| SpecialDay              | float64 | 0.2             | 0.06    | 333.33             | 0.0       | 0.0                    |             0.05 |                   6 |        0    | dummy_sem_null    | -           |
#+-------------------------+---------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+
#| Month                   | object  | -               | -       | -                  | -         | -                      |             0.08 |                  10 |        0    | dummy_sem_null    | -           |
#+-------------------------+---------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+
#| OperatingSystems        | int64   | 0.91            | 2.12    | 42.92              | 2.0       | 94.34                  |             0.06 |                   8 |        0    | dummy_sem_null    | -           |
#+-------------------------+---------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+
#| Browser                 | int64   | 1.72            | 2.36    | 72.88              | 2.0       | 84.75                  |             0.11 |                  13 |        0    | dummy_sem_null    | -           |
#+-------------------------+---------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+
#| Region                  | int64   | 2.4             | 3.15    | 76.19              | 3.0       | 95.24                  |             0.07 |                   9 |        0    | dummy_sem_null    | -           |
#+-------------------------+---------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+
#| TrafficType             | int64   | 4.03            | 4.07    | 99.02              | 2.0       | 49.14                  |             0.16 |                  20 |        0    | dummy_sem_null    | -           |
#+-------------------------+---------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+
#| VisitorType             | object  | -               | -       | -                  | -         | -                      |             0.02 |                   3 |        0    | dummy_sem_null    | -           |
#+-------------------------+---------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+
#| Weekend                 | bool    | -               | -       | -                  | -         | -                      |             0.02 |                   2 |        0    | dummy_sem_null    | -           |
#+-------------------------+---------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+

# Correlacao entre as variaveis:
#+-------------------------+------------------+---------------------------+-----------------+--------------------------+------------------+---------------------------+---------------+-------------+--------------+--------------+--------------------+-----------+----------+---------------+-----------+-----------+
#|                         |   Administrative |   Administrative_Duration |   Informational |   Informational_Duration |   ProductRelated |   ProductRelated_Duration |   BounceRates |   ExitRates |   PageValues |   SpecialDay |   OperatingSystems |   Browser |   Region |   TrafficType |   Weekend |   Revenue |
#+=========================+==================+===========================+=================+==========================+==================+===========================+===============+=============+==============+==============+====================+===========+==========+===============+===========+===========+
#| Administrative          |            1     |                     0.601 |           0.377 |                    0.256 |            0.431 |                     0.374 |        -0.223 |      -0.316 |        0.099 |       -0.095 |             -0.006 |    -0.025 |   -0.006 |        -0.034 |     0.026 |     0.139 |
#+-------------------------+------------------+---------------------------+-----------------+--------------------------+------------------+---------------------------+---------------+-------------+--------------+--------------+--------------------+-----------+----------+---------------+-----------+-----------+
#| Administrative_Duration |            0.601 |                     1     |           0.303 |                    0.238 |            0.289 |                     0.355 |        -0.144 |      -0.206 |        0.067 |       -0.073 |             -0.007 |    -0.016 |   -0.006 |        -0.014 |     0.015 |     0.093 |
#+-------------------------+------------------+---------------------------+-----------------+--------------------------+------------------+---------------------------+---------------+-------------+--------------+--------------+--------------------+-----------+----------+---------------+-----------+-----------+
#| Informational           |            0.377 |                     0.303 |           1     |                    0.619 |            0.374 |                     0.387 |        -0.116 |      -0.164 |        0.049 |       -0.048 |             -0.009 |    -0.038 |   -0.029 |        -0.035 |     0.036 |     0.095 |
#+-------------------------+------------------+---------------------------+-----------------+--------------------------+------------------+---------------------------+---------------+-------------+--------------+--------------+--------------------+-----------+----------+---------------+-----------+-----------+
#| Informational_Duration  |            0.256 |                     0.238 |           0.619 |                    1     |            0.28  |                     0.347 |        -0.074 |      -0.105 |        0.031 |       -0.031 |             -0.01  |    -0.019 |   -0.027 |        -0.025 |     0.024 |     0.07  |
#+-------------------------+------------------+---------------------------+-----------------+--------------------------+------------------+---------------------------+---------------+-------------+--------------+--------------+--------------------+-----------+----------+---------------+-----------+-----------+
#| ProductRelated          |            0.431 |                     0.289 |           0.374 |                    0.28  |            1     |                     0.861 |        -0.204 |      -0.292 |        0.056 |       -0.024 |              0.004 |    -0.013 |   -0.038 |        -0.043 |     0.016 |     0.158 |
#+-------------------------+------------------+---------------------------+-----------------+--------------------------+------------------+---------------------------+---------------+-------------+--------------+--------------+--------------------+-----------+----------+---------------+-----------+-----------+
#| ProductRelated_Duration |            0.374 |                     0.355 |           0.387 |                    0.347 |            0.861 |                     1     |        -0.184 |      -0.252 |        0.053 |       -0.037 |              0.003 |    -0.008 |   -0.033 |        -0.037 |     0.007 |     0.152 |
#+-------------------------+------------------+---------------------------+-----------------+--------------------------+------------------+---------------------------+---------------+-------------+--------------+--------------+--------------------+-----------+----------+---------------+-----------+-----------+
#| BounceRates             |           -0.223 |                    -0.144 |          -0.116 |                   -0.074 |           -0.204 |                    -0.184 |         1     |       0.913 |       -0.119 |        0.073 |              0.024 |    -0.016 |   -0.007 |         0.079 |    -0.047 |    -0.151 |
#+-------------------------+------------------+---------------------------+-----------------+--------------------------+------------------+---------------------------+---------------+-------------+--------------+--------------+--------------------+-----------+----------+---------------+-----------+-----------+
#| ExitRates               |           -0.316 |                    -0.206 |          -0.164 |                   -0.105 |           -0.292 |                    -0.252 |         0.913 |       1     |       -0.174 |        0.103 |              0.015 |    -0.004 |   -0.009 |         0.079 |    -0.063 |    -0.207 |
#+-------------------------+------------------+---------------------------+-----------------+--------------------------+------------------+---------------------------+---------------+-------------+--------------+--------------+--------------------+-----------+----------+---------------+-----------+-----------+
#| PageValues              |            0.099 |                     0.067 |           0.049 |                    0.031 |            0.056 |                     0.053 |        -0.119 |      -0.174 |        1     |       -0.064 |              0.019 |     0.046 |    0.011 |         0.013 |     0.012 |     0.493 |
#+-------------------------+------------------+---------------------------+-----------------+--------------------------+------------------+---------------------------+---------------+-------------+--------------+--------------+--------------------+-----------+----------+---------------+-----------+-----------+
#| SpecialDay              |           -0.095 |                    -0.073 |          -0.048 |                   -0.031 |           -0.024 |                    -0.037 |         0.073 |       0.103 |       -0.064 |        1     |              0.013 |     0.003 |   -0.016 |         0.052 |    -0.017 |    -0.082 |
#+-------------------------+------------------+---------------------------+-----------------+--------------------------+------------------+---------------------------+---------------+-------------+--------------+--------------+--------------------+-----------+----------+---------------+-----------+-----------+
#| OperatingSystems        |           -0.006 |                    -0.007 |          -0.009 |                   -0.01  |            0.004 |                     0.003 |         0.024 |       0.015 |        0.019 |        0.013 |              1     |     0.223 |    0.077 |         0.189 |     0     |    -0.015 |
#+-------------------------+------------------+---------------------------+-----------------+--------------------------+------------------+---------------------------+---------------+-------------+--------------+--------------+--------------------+-----------+----------+---------------+-----------+-----------+
#| Browser                 |           -0.025 |                    -0.016 |          -0.038 |                   -0.019 |           -0.013 |                    -0.008 |        -0.016 |      -0.004 |        0.046 |        0.003 |              0.223 |     1     |    0.097 |         0.112 |    -0.04  |     0.024 |
#+-------------------------+------------------+---------------------------+-----------------+--------------------------+------------------+---------------------------+---------------+-------------+--------------+--------------+--------------------+-----------+----------+---------------+-----------+-----------+
#| Region                  |           -0.006 |                    -0.006 |          -0.029 |                   -0.027 |           -0.038 |                    -0.033 |        -0.007 |      -0.009 |        0.011 |       -0.016 |              0.077 |     0.097 |    1     |         0.048 |    -0.001 |    -0.012 |
#+-------------------------+------------------+---------------------------+-----------------+--------------------------+------------------+---------------------------+---------------+-------------+--------------+--------------+--------------------+-----------+----------+---------------+-----------+-----------+
#| TrafficType             |           -0.034 |                    -0.014 |          -0.035 |                   -0.025 |           -0.043 |                    -0.037 |         0.079 |       0.079 |        0.013 |        0.052 |              0.189 |     0.112 |    0.048 |         1     |    -0.002 |    -0.005 |
#+-------------------------+------------------+---------------------------+-----------------+--------------------------+------------------+---------------------------+---------------+-------------+--------------+--------------+--------------------+-----------+----------+---------------+-----------+-----------+
#| Weekend                 |            0.026 |                     0.015 |           0.036 |                    0.024 |            0.016 |                     0.007 |        -0.047 |      -0.063 |        0.012 |       -0.017 |              0     |    -0.04  |   -0.001 |        -0.002 |     1     |     0.029 |
#+-------------------------+------------------+---------------------------+-----------------+--------------------------+------------------+---------------------------+---------------+-------------+--------------+--------------+--------------------+-----------+----------+---------------+-----------+-----------+
#| Revenue                 |            0.139 |                     0.093 |           0.095 |                    0.07  |            0.158 |                     0.152 |        -0.151 |      -0.207 |        0.493 |       -0.082 |             -0.015 |     0.024 |   -0.012 |        -0.005 |     0.029 |     1     |
#+-------------------------+------------------+---------------------------+-----------------+--------------------------+------------------+---------------------------+---------------+-------------+--------------+--------------+--------------------+-----------+----------+---------------+-----------+-----------+


# Tratamento de Dados
df2 = pd.concat([df['Revenue']])

# Criando Dataframe auxiliar
df_temp = pd.DataFrame()

# Adicionando colunas continuas sem nulos
df2 = pd.concat([df['PageValues'], df2], axis=1)

# Adicionando colunas continuas com nulos
df_temp = df.Administrative_Duration.fillna( df.Administrative_Duration.mean() )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = df.Informational_Duration.fillna( df.Informational_Duration.mean() )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = df.ProductRelated.fillna( df.ProductRelated.mean() )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = df.ProductRelated_Duration.fillna( df.ProductRelated_Duration.mean() )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = df.BounceRates.fillna( df.BounceRates.mean() )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = df.ExitRates.fillna( df.ExitRates.mean() )
df2 = pd.concat([df_temp, df2], axis=1)

# Adicionando e preparando colunas com dummies sem tratamento de NULL
df_temp = pd.get_dummies( df.SpecialDay, prefix='SpecialDay' )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = pd.get_dummies( df.Month, prefix='Month' )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = pd.get_dummies( df.OperatingSystems, prefix='OperatingSystems' )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = pd.get_dummies( df.Browser, prefix='Browser' )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = pd.get_dummies( df.Region, prefix='Region' )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = pd.get_dummies( df.TrafficType, prefix='TrafficType' )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = pd.get_dummies( df.VisitorType, prefix='VisitorType' )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = pd.get_dummies( df.Weekend, prefix='Weekend' )
df2 = pd.concat([df_temp, df2], axis=1)

# Adicionando e preparando colunas com dummies com tratamento de NULL
df_temp = df.Administrative.fillna( 'NULL' )
df_temp = pd.get_dummies( df_temp, prefix='Administrative' )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = df.Informational.fillna( 'NULL' )
df_temp = pd.get_dummies( df_temp, prefix='Informational' )
df2 = pd.concat([df_temp, df2], axis=1)


# Dividindo a base entre variaveis explicativas e variavel resposta
X = df2.loc[:, df2.columns !='Revenue']
Y = df2.Revenue

# Separando a Base entre Teste e Treino
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=0.75, test_size=0.25, random_state=7)


# Modelo: DecisionTreeClassifier
modelo = DecisionTreeClassifier()

# Tunning: 
parametros = {'max_depth': [2, 4, 8, 12, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'max_features': [31, 62, 93, 124]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='accuracy', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'max_depth': 4, 'max_features': 124, 'min_samples_leaf': 5, 'min_samples_split': 10}

# Aplicando o Modelo
model = DecisionTreeClassifier(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(accuracy_score(Y_train, predict_train),6)
score_teste = round(accuracy_score(Y_test, predict_test),6)

# ----------------------------------------------


# Modelo: XGBClassifier
modelo = XGBClassifier()

# Tunning: 
parametros = {'nthread': [4], 'objective': ['binary:logistic'], 'learning_rate': [0.05], 'max_depth': [8, 9, 10], 'min_child_weight': [12, 20, 30], 'silent': [1], 'subsample': [0.5, 0.6, 0.7], 'colsample_bytree': [0.6, 0.7, 0.8], 'n_estimators': [5, 10, 20]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='accuracy', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 8, 'min_child_weight': 30, 'n_estimators': 20, 'nthread': 4, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.7}

# Aplicando o Modelo
model = XGBClassifier(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(accuracy_score(Y_train, predict_train),6)
score_teste = round(accuracy_score(Y_test, predict_test),6)

# ----------------------------------------------


# Modelo: RandomForestClassifier
modelo = RandomForestClassifier()

# Tunning: 
parametros = {'n_estimators': [10, 50, 100], 'max_depth': [2, 8, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'bootstrap': [True, False], 'max_features': [31, 62, 93, 124]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='accuracy', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'bootstrap': True, 'max_depth': 16, 'max_features': 93, 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 100}

# Aplicando o Modelo
model = RandomForestClassifier(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(accuracy_score(Y_train, predict_train),6)
score_teste = round(accuracy_score(Y_test, predict_test),6)

# ----------------------------------------------


# Modelo: GradientBoostingClassifier
modelo = GradientBoostingClassifier()

# Tunning: 
parametros = {'learning_rate': [1, 0.25, 0.05, 0.01], 'n_estimators': [4, 16, 100], 'max_depth': [2, 8, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'max_features': [31, 62, 93, 124]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='accuracy', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'learning_rate': 0.05, 'max_depth': 16, 'max_features': 31, 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 100}

# Aplicando o Modelo
model = GradientBoostingClassifier(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(accuracy_score(Y_train, predict_train),6)
score_teste = round(accuracy_score(Y_test, predict_test),6)

# ----------------------------------------------


# Modelo: KNeighborsClassifier
modelo = KNeighborsClassifier()

# Tunning: 
parametros = {'n_neighbors': [2, 8, 16, 32], 'p': [1, 3, 5]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='accuracy', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'n_neighbors': 8, 'p': 5}

# Aplicando o Modelo
model = KNeighborsClassifier(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(accuracy_score(Y_train, predict_train),6)
score_teste = round(accuracy_score(Y_test, predict_test),6)

# ----------------------------------------------


# Modelo: LogisticRegression
modelo = LogisticRegression()

# Tunning: 
parametros = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='accuracy', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'C': 0.001}

# Aplicando o Modelo
model = LogisticRegression(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(accuracy_score(Y_train, predict_train),6)
score_teste = round(accuracy_score(Y_test, predict_test),6)

# ----------------------------------------------


# Modelo: SVC
modelo = SVC()

# Tunning: 
parametros = {'gamma': [0.05, 0.1, 1, 5, 10], 'C': [0.1, 1, 10, 100, 1000], 'degree': [0, 1, 2, 3, 5, 7]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='accuracy', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'C': 1, 'degree': 0, 'gamma': 0.05}

# Aplicando o Modelo
model = SVC(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(accuracy_score(Y_train, predict_train),6)
score_teste = round(accuracy_score(Y_test, predict_test),6)

# ----------------------------------------------


# Modelo: DecisionTreeClassifier
modelo = DecisionTreeClassifier()

# Tunning: 
parametros = {'max_depth': [2, 4, 8, 12, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'max_features': [31, 62, 93, 124]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='f1', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'max_depth': 2, 'max_features': 93, 'min_samples_leaf': 5, 'min_samples_split': 1.0}

# Aplicando o Modelo
model = DecisionTreeClassifier(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(f1_score(Y_train, predict_train),6)
score_teste = round(f1_score(Y_test, predict_test),6)

# ----------------------------------------------


# Modelo: XGBClassifier
modelo = XGBClassifier()

# Tunning: 
parametros = {'nthread': [4], 'objective': ['binary:logistic'], 'learning_rate': [0.05], 'max_depth': [8, 9, 10], 'min_child_weight': [12, 20, 30], 'silent': [1], 'subsample': [0.5, 0.6, 0.7], 'colsample_bytree': [0.6, 0.7, 0.8], 'n_estimators': [5, 10, 20]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='f1', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 8, 'min_child_weight': 30, 'n_estimators': 20, 'nthread': 4, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.7}

# Aplicando o Modelo
model = XGBClassifier(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(f1_score(Y_train, predict_train),6)
score_teste = round(f1_score(Y_test, predict_test),6)

# ----------------------------------------------


# Modelo: RandomForestClassifier
modelo = RandomForestClassifier()

# Tunning: 
parametros = {'n_estimators': [10, 50, 100], 'max_depth': [2, 8, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'bootstrap': [True, False], 'max_features': [31, 62, 93, 124]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='f1', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'bootstrap': False, 'max_depth': 16, 'max_features': 62, 'min_samples_leaf': 5, 'min_samples_split': 0.1, 'n_estimators': 100}

# Aplicando o Modelo
model = RandomForestClassifier(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(f1_score(Y_train, predict_train),6)
score_teste = round(f1_score(Y_test, predict_test),6)

# ----------------------------------------------


# Modelo: GradientBoostingClassifier
modelo = GradientBoostingClassifier()

# Tunning: 
parametros = {'learning_rate': [1, 0.25, 0.05, 0.01], 'n_estimators': [4, 16, 100], 'max_depth': [2, 8, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'max_features': [31, 62, 93, 124]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='f1', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'learning_rate': 0.25, 'max_depth': 2, 'max_features': 124, 'min_samples_leaf': 0.1, 'min_samples_split': 0.1, 'n_estimators': 100}

# Aplicando o Modelo
model = GradientBoostingClassifier(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(f1_score(Y_train, predict_train),6)
score_teste = round(f1_score(Y_test, predict_test),6)

# ----------------------------------------------


# Modelo: KNeighborsClassifier
modelo = KNeighborsClassifier()

# Tunning: 
parametros = {'n_neighbors': [2, 8, 16, 32], 'p': [1, 3, 5]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='f1', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'n_neighbors': 8, 'p': 5}

# Aplicando o Modelo
model = KNeighborsClassifier(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(f1_score(Y_train, predict_train),6)
score_teste = round(f1_score(Y_test, predict_test),6)

# ----------------------------------------------


# Modelo: LogisticRegression
modelo = LogisticRegression()

# Tunning: 
parametros = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='f1', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'C': 1}

# Aplicando o Modelo
model = LogisticRegression(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(f1_score(Y_train, predict_train),6)
score_teste = round(f1_score(Y_test, predict_test),6)

# ----------------------------------------------


# Modelo: SVC
modelo = SVC()

# Tunning: 
parametros = {'gamma': [0.05, 0.1, 1, 5, 10], 'C': [0.1, 1, 10, 100, 1000], 'degree': [0, 1, 2, 3, 5, 7]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='f1', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'C': 10, 'degree': 0, 'gamma': 0.05}

# Aplicando o Modelo
model = SVC(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(f1_score(Y_train, predict_train),6)
score_teste = round(f1_score(Y_test, predict_test),6)

# ----------------------------------------------


# Modelo: DecisionTreeClassifier
modelo = DecisionTreeClassifier()

# Tunning: 
parametros = {'max_depth': [2, 4, 8, 12, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'max_features': [31, 62, 93, 124]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='precision', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'max_depth': 2, 'max_features': 124, 'min_samples_leaf': 5, 'min_samples_split': 0.1}

# Aplicando o Modelo
model = DecisionTreeClassifier(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(precision_score(Y_train, predict_train),6)
score_teste = round(precision_score(Y_test, predict_test),6)

# ----------------------------------------------


# Modelo: XGBClassifier
modelo = XGBClassifier()

# Tunning: 
parametros = {'nthread': [4], 'objective': ['binary:logistic'], 'learning_rate': [0.05], 'max_depth': [8, 9, 10], 'min_child_weight': [12, 20, 30], 'silent': [1], 'subsample': [0.5, 0.6, 0.7], 'colsample_bytree': [0.6, 0.7, 0.8], 'n_estimators': [5, 10, 20]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='precision', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'max_depth': 9, 'min_child_weight': 20, 'n_estimators': 20, 'nthread': 4, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.5}

# Aplicando o Modelo
model = XGBClassifier(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(precision_score(Y_train, predict_train),6)
score_teste = round(precision_score(Y_test, predict_test),6)

# ----------------------------------------------


# Modelo: RandomForestClassifier
modelo = RandomForestClassifier()

# Tunning: 
parametros = {'n_estimators': [10, 50, 100], 'max_depth': [2, 8, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'bootstrap': [True, False], 'max_features': [31, 62, 93, 124]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='precision', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'bootstrap': True, 'max_depth': 2, 'max_features': 62, 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 10}

# Aplicando o Modelo
model = RandomForestClassifier(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(precision_score(Y_train, predict_train),6)
score_teste = round(precision_score(Y_test, predict_test),6)

# ----------------------------------------------


# Modelo: GradientBoostingClassifier
modelo = GradientBoostingClassifier()

# Tunning: 
parametros = {'learning_rate': [1, 0.25, 0.05, 0.01], 'n_estimators': [4, 16, 100], 'max_depth': [2, 8, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'max_features': [31, 62, 93, 124]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='precision', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'learning_rate': 0.05, 'max_depth': 16, 'max_features': 31, 'min_samples_leaf': 5, 'min_samples_split': 0.1, 'n_estimators': 16}

# Aplicando o Modelo
model = GradientBoostingClassifier(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(precision_score(Y_train, predict_train),6)
score_teste = round(precision_score(Y_test, predict_test),6)

# ----------------------------------------------


# Modelo: KNeighborsClassifier
modelo = KNeighborsClassifier()

# Tunning: 
parametros = {'n_neighbors': [2, 8, 16, 32], 'p': [1, 3, 5]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='precision', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'n_neighbors': 32, 'p': 5}

# Aplicando o Modelo
model = KNeighborsClassifier(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(precision_score(Y_train, predict_train),6)
score_teste = round(precision_score(Y_test, predict_test),6)

# ----------------------------------------------


# Modelo: LogisticRegression
modelo = LogisticRegression()

# Tunning: 
parametros = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='precision', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'C': 0.001}

# Aplicando o Modelo
model = LogisticRegression(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(precision_score(Y_train, predict_train),6)
score_teste = round(precision_score(Y_test, predict_test),6)

# ----------------------------------------------


# Modelo: SVC
modelo = SVC()

# Tunning: 
parametros = {'gamma': [0.05, 0.1, 1, 5, 10], 'C': [0.1, 1, 10, 100, 1000], 'degree': [0, 1, 2, 3, 5, 7]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='precision', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'C': 10, 'degree': 0, 'gamma': 0.05}

# Aplicando o Modelo
model = SVC(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(precision_score(Y_train, predict_train),6)
score_teste = round(precision_score(Y_test, predict_test),6)

# ----------------------------------------------


# Modelo: DecisionTreeClassifier
modelo = DecisionTreeClassifier()

# Tunning: 
parametros = {'max_depth': [2, 4, 8, 12, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'max_features': [31, 62, 93, 124]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='recall', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'max_depth': 2, 'max_features': 93, 'min_samples_leaf': 0.1, 'min_samples_split': 1.0}

# Aplicando o Modelo
model = DecisionTreeClassifier(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(recall_score(Y_train, predict_train),6)
score_teste = round(recall_score(Y_test, predict_test),6)

# ----------------------------------------------


# Modelo: XGBClassifier
modelo = XGBClassifier()

# Tunning: 
parametros = {'nthread': [4], 'objective': ['binary:logistic'], 'learning_rate': [0.05], 'max_depth': [8, 9, 10], 'min_child_weight': [12, 20, 30], 'silent': [1], 'subsample': [0.5, 0.6, 0.7], 'colsample_bytree': [0.6, 0.7, 0.8], 'n_estimators': [5, 10, 20]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='recall', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 8, 'min_child_weight': 30, 'n_estimators': 20, 'nthread': 4, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.7}

# Aplicando o Modelo
model = XGBClassifier(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(recall_score(Y_train, predict_train),6)
score_teste = round(recall_score(Y_test, predict_test),6)

# ----------------------------------------------


# Modelo: RandomForestClassifier
modelo = RandomForestClassifier()

# Tunning: 
parametros = {'n_estimators': [10, 50, 100], 'max_depth': [2, 8, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'bootstrap': [True, False], 'max_features': [31, 62, 93, 124]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='recall', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'bootstrap': False, 'max_depth': 2, 'max_features': 124, 'min_samples_leaf': 0.1, 'min_samples_split': 1.0, 'n_estimators': 10}

# Aplicando o Modelo
model = RandomForestClassifier(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(recall_score(Y_train, predict_train),6)
score_teste = round(recall_score(Y_test, predict_test),6)

# ----------------------------------------------


# Modelo: GradientBoostingClassifier
modelo = GradientBoostingClassifier()

# Tunning: 
parametros = {'learning_rate': [1, 0.25, 0.05, 0.01], 'n_estimators': [4, 16, 100], 'max_depth': [2, 8, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'max_features': [31, 62, 93, 124]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='recall', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'learning_rate': 1, 'max_depth': 16, 'max_features': 93, 'min_samples_leaf': 0.1, 'min_samples_split': 1.0, 'n_estimators': 4}

# Aplicando o Modelo
model = GradientBoostingClassifier(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(recall_score(Y_train, predict_train),6)
score_teste = round(recall_score(Y_test, predict_test),6)

# ----------------------------------------------


# Modelo: KNeighborsClassifier
modelo = KNeighborsClassifier()

# Tunning: 
parametros = {'n_neighbors': [2, 8, 16, 32], 'p': [1, 3, 5]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='recall', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'n_neighbors': 2, 'p': 5}

# Aplicando o Modelo
model = KNeighborsClassifier(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(recall_score(Y_train, predict_train),6)
score_teste = round(recall_score(Y_test, predict_test),6)

# ----------------------------------------------


# Modelo: LogisticRegression
modelo = LogisticRegression()

# Tunning: 
parametros = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='recall', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'C': 1}

# Aplicando o Modelo
model = LogisticRegression(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(recall_score(Y_train, predict_train),6)
score_teste = round(recall_score(Y_test, predict_test),6)

# ----------------------------------------------


# Modelo: SVC
modelo = SVC()

# Tunning: 
parametros = {'gamma': [0.05, 0.1, 1, 5, 10], 'C': [0.1, 1, 10, 100, 1000], 'degree': [0, 1, 2, 3, 5, 7]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='recall', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'C': 10, 'degree': 0, 'gamma': 0.05}

# Aplicando o Modelo
model = SVC(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(recall_score(Y_train, predict_train),6)
score_teste = round(recall_score(Y_test, predict_test),6)

# ----------------------------------------------


# Modelo: DecisionTreeClassifier
modelo = DecisionTreeClassifier()

# Tunning: 
parametros = {'max_depth': [2, 4, 8, 12, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'max_features': [31, 62, 93, 124]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='roc_auc', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'max_depth': 12, 'max_features': 124, 'min_samples_leaf': 5, 'min_samples_split': 0.1}

# Aplicando o Modelo
model = DecisionTreeClassifier(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(roc_auc_score(Y_train, predict_train),6)
score_teste = round(roc_auc_score(Y_test, predict_test),6)

# ----------------------------------------------


# Modelo: XGBClassifier
modelo = XGBClassifier()

# Tunning: 
parametros = {'nthread': [4], 'objective': ['binary:logistic'], 'learning_rate': [0.05], 'max_depth': [8, 9, 10], 'min_child_weight': [12, 20, 30], 'silent': [1], 'subsample': [0.5, 0.6, 0.7], 'colsample_bytree': [0.6, 0.7, 0.8], 'n_estimators': [5, 10, 20]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='roc_auc', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 9, 'min_child_weight': 30, 'n_estimators': 20, 'nthread': 4, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.7}

# Aplicando o Modelo
model = XGBClassifier(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(roc_auc_score(Y_train, predict_train),6)
score_teste = round(roc_auc_score(Y_test, predict_test),6)

# ----------------------------------------------


# Modelo: RandomForestClassifier
modelo = RandomForestClassifier()

# Tunning: 
parametros = {'n_estimators': [10, 50, 100], 'max_depth': [2, 8, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'bootstrap': [True, False], 'max_features': [31, 62, 93, 124]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='roc_auc', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'bootstrap': False, 'max_depth': 8, 'max_features': 62, 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 100}

# Aplicando o Modelo
model = RandomForestClassifier(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(roc_auc_score(Y_train, predict_train),6)
score_teste = round(roc_auc_score(Y_test, predict_test),6)

# ----------------------------------------------


# Modelo: GradientBoostingClassifier
modelo = GradientBoostingClassifier()

# Tunning: 
parametros = {'learning_rate': [1, 0.25, 0.05, 0.01], 'n_estimators': [4, 16, 100], 'max_depth': [2, 8, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'max_features': [31, 62, 93, 124]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='roc_auc', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'learning_rate': 0.05, 'max_depth': 8, 'max_features': 31, 'min_samples_leaf': 5, 'min_samples_split': 0.1, 'n_estimators': 100}

# Aplicando o Modelo
model = GradientBoostingClassifier(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(roc_auc_score(Y_train, predict_train),6)
score_teste = round(roc_auc_score(Y_test, predict_test),6)

# ----------------------------------------------


# Modelo: KNeighborsClassifier
modelo = KNeighborsClassifier()

# Tunning: 
parametros = {'n_neighbors': [2, 8, 16, 32], 'p': [1, 3, 5]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='roc_auc', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'n_neighbors': 32, 'p': 1}

# Aplicando o Modelo
model = KNeighborsClassifier(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(roc_auc_score(Y_train, predict_train),6)
score_teste = round(roc_auc_score(Y_test, predict_test),6)

# ----------------------------------------------


# Modelo: LogisticRegression
modelo = LogisticRegression()

# Tunning: 
parametros = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='roc_auc', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'C': 0.01}

# Aplicando o Modelo
model = LogisticRegression(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(roc_auc_score(Y_train, predict_train),6)
score_teste = round(roc_auc_score(Y_test, predict_test),6)

# ----------------------------------------------


# Modelo: SVC
modelo = SVC()

# Tunning: 
parametros = {'gamma': [0.05, 0.1, 1, 5, 10], 'C': [0.1, 1, 10, 100, 1000], 'degree': [0, 1, 2, 3, 5, 7]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='roc_auc', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'C': 1, 'degree': 0, 'gamma': 0.05}

# Aplicando o Modelo
model = SVC(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(roc_auc_score(Y_train, predict_train),6)
score_teste = round(roc_auc_score(Y_test, predict_test),6)

# ----------------------------------------------

# Comparativo de Modelos: 

#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| Modelo                     | Metrica_Score   |   Tempo_Exec(s) |   Score_Treino |   Score_Teste | Melhor_hiper_param                                                                                                                                                                        |
#+============================+=================+=================+================+===============+===========================================================================================================================================================================================+
#| DecisionTreeClassifier     | accuracy        |           20.43 |       0.906889 |      0.903665 | {'max_depth': 4, 'max_features': 124, 'min_samples_leaf': 5, 'min_samples_split': 10}                                                                                                     |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| XGBClassifier              | accuracy        |          331.95 |       0.908619 |      0.904314 | {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 8, 'min_child_weight': 30, 'n_estimators': 20, 'nthread': 4, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.7} |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| RandomForestClassifier     | accuracy        |          644.96 |       0.955445 |      0.907233 | {'bootstrap': True, 'max_depth': 16, 'max_features': 93, 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 100}                                                             |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| GradientBoostingClassifier | accuracy        |         2149.68 |       1        |      0.90626  | {'learning_rate': 0.05, 'max_depth': 16, 'max_features': 31, 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 100}                                                         |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| KNeighborsClassifier       | accuracy        |           90.83 |       0.879637 |      0.869283 | {'n_neighbors': 8, 'p': 5}                                                                                                                                                                |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| LogisticRegression         | accuracy        |            2.32 |       0.883097 |      0.885825 | {'C': 0.001}                                                                                                                                                                              |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| SVC                        | accuracy        |         6249.53 |       0.994485 |      0.842361 | {'C': 1, 'degree': 0, 'gamma': 0.05}                                                                                                                                                      |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| DecisionTreeClassifier     | f1              |           16.27 |       0.657526 |      0.688153 | {'max_depth': 2, 'max_features': 93, 'min_samples_leaf': 5, 'min_samples_split': 1.0}                                                                                                     |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| XGBClassifier              | f1              |          328.78 |       0.670051 |      0.658169 | {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 8, 'min_child_weight': 30, 'n_estimators': 20, 'nthread': 4, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.7} |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| RandomForestClassifier     | f1              |          662.88 |       0.668624 |      0.655629 | {'bootstrap': False, 'max_depth': 16, 'max_features': 62, 'min_samples_leaf': 5, 'min_samples_split': 0.1, 'n_estimators': 100}                                                           |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| GradientBoostingClassifier | f1              |         1947.83 |       0.679502 |      0.674888 | {'learning_rate': 0.25, 'max_depth': 2, 'max_features': 124, 'min_samples_leaf': 0.1, 'min_samples_split': 0.1, 'n_estimators': 100}                                                      |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| KNeighborsClassifier       | f1              |           87.61 |       0.417582 |      0.379045 | {'n_neighbors': 8, 'p': 5}                                                                                                                                                                |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| LogisticRegression         | f1              |            2.35 |       0.507186 |      0.508796 | {'C': 1}                                                                                                                                                                                  |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| SVC                        | f1              |         6286.89 |       0.999648 |      0.008016 | {'C': 10, 'degree': 0, 'gamma': 0.05}                                                                                                                                                     |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| DecisionTreeClassifier     | precision       |           13.67 |       0.729761 |      0.748387 | {'max_depth': 2, 'max_features': 124, 'min_samples_leaf': 5, 'min_samples_split': 0.1}                                                                                                    |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| XGBClassifier              | precision       |          336.98 |       0.807735 |      0.83274  | {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'max_depth': 9, 'min_child_weight': 20, 'n_estimators': 20, 'nthread': 4, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.5} |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| RandomForestClassifier     | precision       |          651.44 |       0.825832 |      0.834225 | {'bootstrap': True, 'max_depth': 2, 'max_features': 62, 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 10}                                                               |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| GradientBoostingClassifier | precision       |         1978.6  |       0.953271 |      0.942857 | {'learning_rate': 0.05, 'max_depth': 16, 'max_features': 31, 'min_samples_leaf': 5, 'min_samples_split': 0.1, 'n_estimators': 16}                                                         |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| KNeighborsClassifier       | precision       |          105.54 |       0.878788 |      0.894737 | {'n_neighbors': 32, 'p': 5}                                                                                                                                                               |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| LogisticRegression         | precision       |            2.96 |       0.753343 |      0.768    | {'C': 0.001}                                                                                                                                                                              |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| SVC                        | precision       |         6270.54 |       1        |      0.153846 | {'C': 10, 'degree': 0, 'gamma': 0.05}                                                                                                                                                     |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| DecisionTreeClassifier     | recall          |           13.01 |       0.792546 |      0.812757 | {'max_depth': 2, 'max_features': 93, 'min_samples_leaf': 0.1, 'min_samples_split': 1.0}                                                                                                   |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| XGBClassifier              | recall          |          332.63 |       0.603376 |      0.584362 | {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 8, 'min_child_weight': 30, 'n_estimators': 20, 'nthread': 4, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.7} |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| RandomForestClassifier     | recall          |          668.3  |       0.792546 |      0.812757 | {'bootstrap': False, 'max_depth': 2, 'max_features': 124, 'min_samples_leaf': 0.1, 'min_samples_split': 1.0, 'n_estimators': 10}                                                          |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| GradientBoostingClassifier | recall          |         1909.52 |       0.555556 |      0.555556 | {'learning_rate': 1, 'max_depth': 16, 'max_features': 93, 'min_samples_leaf': 0.1, 'min_samples_split': 1.0, 'n_estimators': 4}                                                           |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| KNeighborsClassifier       | recall          |           91.23 |       0.396624 |      0.246914 | {'n_neighbors': 2, 'p': 5}                                                                                                                                                                |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| LogisticRegression         | recall          |            2.36 |       0.384669 |      0.386831 | {'C': 1}                                                                                                                                                                                  |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| SVC                        | recall          |         6369.5  |       0.999297 |      0.004115 | {'C': 10, 'degree': 0, 'gamma': 0.05}                                                                                                                                                     |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| DecisionTreeClassifier     | roc_auc         |           17.36 |       0.782617 |      0.781164 | {'max_depth': 12, 'max_features': 124, 'min_samples_leaf': 5, 'min_samples_split': 0.1}                                                                                                   |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| XGBClassifier              | roc_auc         |          336.74 |       0.783796 |      0.774276 | {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 9, 'min_child_weight': 30, 'n_estimators': 20, 'nthread': 4, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.7} |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| RandomForestClassifier     | roc_auc         |          663.34 |       0.835673 |      0.782181 | {'bootstrap': False, 'max_depth': 8, 'max_features': 62, 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 100}                                                             |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| GradientBoostingClassifier | roc_auc         |         1960.67 |       0.808569 |      0.78688  | {'learning_rate': 0.05, 'max_depth': 8, 'max_features': 31, 'min_samples_leaf': 5, 'min_samples_split': 0.1, 'n_estimators': 100}                                                         |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| KNeighborsClassifier       | roc_auc         |           83.11 |       0.532381 |      0.525786 | {'n_neighbors': 32, 'p': 1}                                                                                                                                                               |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| LogisticRegression         | roc_auc         |            2.43 |       0.672554 |      0.688037 | {'C': 0.01}                                                                                                                                                                               |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| SVC                        | roc_auc         |         6593.62 |       0.982068 |      0.500836 | {'C': 1, 'degree': 0, 'gamma': 0.05}                                                                                                                                                      |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
