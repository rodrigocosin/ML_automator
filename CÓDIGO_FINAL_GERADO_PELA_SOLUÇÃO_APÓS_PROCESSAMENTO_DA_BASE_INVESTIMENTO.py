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
target = 'y'

# Importando os dados de entrada
df = pd.read_csv('C:/Users/Rodrigo/Desktop/TCC/bank_customer_survey/bank_customer_survey.csv', sep = ',', decimal = '.')

# Analise de variaveis e identificacao de tratamento:
#+-----------+--------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+
#| Coluna    | Tipo   | Desvio Padrao   | Media   | Coef de Variacao   | Mediana   | Coef de Centralidade   |   Perc_Distintos |   Valores_Distintos |   Perc_Null | Acao              | Desc Acao   |
#+===========+========+=================+=========+====================+===========+========================+==================+=====================+=============+===================+=============+
#| age       | int64  | 10.62           | 40.94   | 25.94              | 39.0      | 95.26                  |             0.17 |                  77 |           0 | continua_sem_null | -           |
#+-----------+--------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+
#| job       | object | -               | -       | -                  | -         | -                      |             0.03 |                  12 |           0 | dummy_sem_null    | -           |
#+-----------+--------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+
#| marital   | object | -               | -       | -                  | -         | -                      |             0.01 |                   3 |           0 | dummy_sem_null    | -           |
#+-----------+--------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+
#| education | object | -               | -       | -                  | -         | -                      |             0.01 |                   4 |           0 | dummy_sem_null    | -           |
#+-----------+--------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+
#| default   | object | -               | -       | -                  | -         | -                      |             0    |                   2 |           0 | dummy_sem_null    | -           |
#+-----------+--------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+
#| balance   | int64  | 3044.77         | 1362.27 | 223.51             | 448.0     | 32.89                  |            15.85 |                7168 |           0 | continua_sem_null | -           |
#+-----------+--------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+
#| housing   | object | -               | -       | -                  | -         | -                      |             0    |                   2 |           0 | dummy_sem_null    | -           |
#+-----------+--------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+
#| loan      | object | -               | -       | -                  | -         | -                      |             0    |                   2 |           0 | dummy_sem_null    | -           |
#+-----------+--------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+
#| contact   | object | -               | -       | -                  | -         | -                      |             0.01 |                   3 |           0 | dummy_sem_null    | -           |
#+-----------+--------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+
#| day       | int64  | 8.32            | 15.81   | 52.62              | 16.0      | 101.2                  |             0.07 |                  31 |           0 | continua_sem_null | -           |
#+-----------+--------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+
#| month     | object | -               | -       | -                  | -         | -                      |             0.03 |                  12 |           0 | dummy_sem_null    | -           |
#+-----------+--------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+
#| duration  | int64  | 257.53          | 258.16  | 99.76              | 180.0     | 69.72                  |             3.48 |                1573 |           0 | continua_sem_null | -           |
#+-----------+--------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+
#| campaign  | int64  | 3.1             | 2.76    | 112.32             | 2.0       | 72.46                  |             0.11 |                  48 |           0 | continua_sem_null | -           |
#+-----------+--------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+
#| pdays     | int64  | 100.13          | 40.2    | 249.08             | -1.0      | -2.49                  |             1.24 |                 559 |           0 | continua_sem_null | -           |
#+-----------+--------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+
#| previous  | int64  | 2.3             | 0.58    | 396.55             | 0.0       | 0.0                    |             0.09 |                  41 |           0 | continua_sem_null | -           |
#+-----------+--------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+
#| poutcome  | object | -               | -       | -                  | -         | -                      |             0.01 |                   4 |           0 | dummy_sem_null    | -           |
#+-----------+--------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+-------------+

# Correlacao entre as variaveis:
#+----------+--------+-----------+--------+------------+------------+---------+------------+--------+
#|          |    age |   balance |    day |   duration |   campaign |   pdays |   previous |      y |
#+==========+========+===========+========+============+============+=========+============+========+
#| age      |  1     |     0.098 | -0.009 |     -0.005 |      0.005 |  -0.024 |      0.001 |  0.025 |
#+----------+--------+-----------+--------+------------+------------+---------+------------+--------+
#| balance  |  0.098 |     1     |  0.005 |      0.022 |     -0.015 |   0.003 |      0.017 |  0.053 |
#+----------+--------+-----------+--------+------------+------------+---------+------------+--------+
#| day      | -0.009 |     0.005 |  1     |     -0.03  |      0.162 |  -0.093 |     -0.052 | -0.028 |
#+----------+--------+-----------+--------+------------+------------+---------+------------+--------+
#| duration | -0.005 |     0.022 | -0.03  |      1     |     -0.085 |  -0.002 |      0.001 |  0.395 |
#+----------+--------+-----------+--------+------------+------------+---------+------------+--------+
#| campaign |  0.005 |    -0.015 |  0.162 |     -0.085 |      1     |  -0.089 |     -0.033 | -0.073 |
#+----------+--------+-----------+--------+------------+------------+---------+------------+--------+
#| pdays    | -0.024 |     0.003 | -0.093 |     -0.002 |     -0.089 |   1     |      0.455 |  0.104 |
#+----------+--------+-----------+--------+------------+------------+---------+------------+--------+
#| previous |  0.001 |     0.017 | -0.052 |      0.001 |     -0.033 |   0.455 |      1     |  0.093 |
#+----------+--------+-----------+--------+------------+------------+---------+------------+--------+
#| y        |  0.025 |     0.053 | -0.028 |      0.395 |     -0.073 |   0.104 |      0.093 |  1     |
#+----------+--------+-----------+--------+------------+------------+---------+------------+--------+


# Tratamento de Dados
df2 = pd.concat([df['y']])

# Criando Dataframe auxiliar
df_temp = pd.DataFrame()

# Adicionando colunas continuas sem nulos
df2 = pd.concat([df['age'], df2], axis=1)
df2 = pd.concat([df['balance'], df2], axis=1)
df2 = pd.concat([df['day'], df2], axis=1)
df2 = pd.concat([df['duration'], df2], axis=1)
df2 = pd.concat([df['campaign'], df2], axis=1)
df2 = pd.concat([df['pdays'], df2], axis=1)
df2 = pd.concat([df['previous'], df2], axis=1)

# Adicionando colunas continuas com nulos

# Adicionando e preparando colunas com dummies sem tratamento de NULL
df_temp = pd.get_dummies( df.job, prefix='job' )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = pd.get_dummies( df.marital, prefix='marital' )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = pd.get_dummies( df.education, prefix='education' )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = pd.get_dummies( df.default, prefix='default' )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = pd.get_dummies( df.housing, prefix='housing' )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = pd.get_dummies( df.loan, prefix='loan' )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = pd.get_dummies( df.contact, prefix='contact' )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = pd.get_dummies( df.month, prefix='month' )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = pd.get_dummies( df.poutcome, prefix='poutcome' )
df2 = pd.concat([df_temp, df2], axis=1)

# Adicionando e preparando colunas com dummies com tratamento de NULL


# Dividindo a base entre variaveis explicativas e variavel resposta
X = df2.loc[:, df2.columns !='y']
Y = df2.y

# Separando a Base entre Teste e Treino
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=0.75, test_size=0.25, random_state=7)


# Modelo: DecisionTreeClassifier
modelo = DecisionTreeClassifier()

# Tunning: 
parametros = {'max_depth': [2, 4, 8, 12, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'max_features': [12, 25, 38, 51]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='accuracy', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'max_depth': 8, 'max_features': 51, 'min_samples_leaf': 5, 'min_samples_split': 10}

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

# Melhores Parametros: {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 10, 'min_child_weight': 12, 'n_estimators': 20, 'nthread': 4, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.5}

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
parametros = {'n_estimators': [10, 50, 100], 'max_depth': [2, 8, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'bootstrap': [True, False], 'max_features': [12, 25, 38, 51]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='accuracy', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'bootstrap': False, 'max_depth': 16, 'max_features': 12, 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 100}

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
parametros = {'learning_rate': [1, 0.25, 0.01], 'n_estimators': [4, 30], 'max_depth': [2, 8, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'max_features': [12, 25, 38, 51]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='accuracy', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'learning_rate': 0.25, 'max_depth': 16, 'max_features': 12, 'min_samples_leaf': 5, 'min_samples_split': 0.1, 'n_estimators': 30}

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

# Melhores Parametros: {'n_neighbors': 32, 'p': 5}

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

# Melhores Parametros: {'C': 1}

# Aplicando o Modelo
model = LogisticRegression(**mod.best_params_)
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
parametros = {'max_depth': [2, 4, 8, 12, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'max_features': [12, 25, 38, 51]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='f1', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'max_depth': 12, 'max_features': 38, 'min_samples_leaf': 5, 'min_samples_split': 10}

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

# Melhores Parametros: {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 10, 'min_child_weight': 12, 'n_estimators': 20, 'nthread': 4, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.7}

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
parametros = {'n_estimators': [10, 50, 100], 'max_depth': [2, 8, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'bootstrap': [True, False], 'max_features': [12, 25, 38, 51]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='f1', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'bootstrap': True, 'max_depth': 16, 'max_features': 51, 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 50}

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
parametros = {'learning_rate': [1, 0.25, 0.01], 'n_estimators': [4, 30], 'max_depth': [2, 8, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'max_features': [12, 25, 38, 51]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='f1', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'learning_rate': 0.25, 'max_depth': 8, 'max_features': 25, 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 30}

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

# Melhores Parametros: {'n_neighbors': 8, 'p': 1}

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


# Modelo: DecisionTreeClassifier
modelo = DecisionTreeClassifier()

# Tunning: 
parametros = {'max_depth': [2, 4, 8, 12, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'max_features': [12, 25, 38, 51]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='precision', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'max_depth': 2, 'max_features': 38, 'min_samples_leaf': 5, 'min_samples_split': 10}

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

# Melhores Parametros: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'max_depth': 10, 'min_child_weight': 20, 'n_estimators': 20, 'nthread': 4, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.7}

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
parametros = {'n_estimators': [10, 50, 100], 'max_depth': [2, 8, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'bootstrap': [True, False], 'max_features': [12, 25, 38, 51]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='precision', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'bootstrap': False, 'max_depth': 2, 'max_features': 12, 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 100}

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
parametros = {'learning_rate': [1, 0.25, 0.01], 'n_estimators': [4, 30], 'max_depth': [2, 8, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'max_features': [12, 25, 38, 51]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='precision', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'learning_rate': 0.25, 'max_depth': 2, 'max_features': 38, 'min_samples_leaf': 5, 'min_samples_split': 1.0, 'n_estimators': 4}

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

# Melhores Parametros: {'C': 0.01}

# Aplicando o Modelo
model = LogisticRegression(**mod.best_params_)
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
parametros = {'max_depth': [2, 4, 8, 12, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'max_features': [12, 25, 38, 51]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='recall', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'max_depth': 16, 'max_features': 25, 'min_samples_leaf': 5, 'min_samples_split': 10}

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

# Melhores Parametros: {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 10, 'min_child_weight': 12, 'n_estimators': 20, 'nthread': 4, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.7}

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
parametros = {'n_estimators': [10, 50, 100], 'max_depth': [2, 8, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'bootstrap': [True, False], 'max_features': [12, 25, 38, 51]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='recall', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'bootstrap': False, 'max_depth': 16, 'max_features': 25, 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 100}

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
parametros = {'learning_rate': [1, 0.25, 0.01], 'n_estimators': [4, 30], 'max_depth': [2, 8, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'max_features': [12, 25, 38, 51]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='recall', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'learning_rate': 1, 'max_depth': 16, 'max_features': 25, 'min_samples_leaf': 5, 'min_samples_split': 0.1, 'n_estimators': 30}

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

# Melhores Parametros: {'n_neighbors': 8, 'p': 5}

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


# Modelo: DecisionTreeClassifier
modelo = DecisionTreeClassifier()

# Tunning: 
parametros = {'max_depth': [2, 4, 8, 12, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'max_features': [12, 25, 38, 51]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='roc_auc', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'max_depth': 16, 'max_features': 51, 'min_samples_leaf': 5, 'min_samples_split': 0.1}

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

# Melhores Parametros: {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 10, 'min_child_weight': 12, 'n_estimators': 20, 'nthread': 4, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.7}

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
parametros = {'n_estimators': [10, 50, 100], 'max_depth': [2, 8, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'bootstrap': [True, False], 'max_features': [12, 25, 38, 51]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='roc_auc', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'bootstrap': True, 'max_depth': 16, 'max_features': 12, 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 100}

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
parametros = {'learning_rate': [1, 0.25, 0.01], 'n_estimators': [4, 30], 'max_depth': [2, 8, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'max_features': [12, 25, 38, 51]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='roc_auc', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'learning_rate': 0.25, 'max_depth': 16, 'max_features': 25, 'min_samples_leaf': 5, 'min_samples_split': 0.1, 'n_estimators': 30}

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

# Melhores Parametros: {'C': 0.1}

# Aplicando o Modelo
model = LogisticRegression(**mod.best_params_)
model.fit(X_train, Y_train)
predict_train = model.predict(X_train)
predict_test = model.predict(X_test)

# Calculando o Score
score_treino = round(roc_auc_score(Y_train, predict_train),6)
score_teste = round(roc_auc_score(Y_test, predict_test),6)

# ----------------------------------------------

# Comparativo de Modelos: 

#+----------------------------+-----------------+-----------------+----------------+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| Modelo                     | Metrica_Score   |   Tempo_Exec(s) |   Score_Treino |   Score_Teste | Melhor_hiper_param                                                                                                                                                                         |
#+============================+=================+=================+================+===============+============================================================================================================================================================================================+
#| DecisionTreeClassifier     | accuracy        |          147.06 |       0.914121 |      0.9033   | {'max_depth': 8, 'max_features': 51, 'min_samples_leaf': 5, 'min_samples_split': 10}                                                                                                       |
#+----------------------------+-----------------+-----------------+----------------+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| XGBClassifier              | accuracy        |         1130.71 |       0.906541 |      0.905246 | {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 10, 'min_child_weight': 12, 'n_estimators': 20, 'nthread': 4, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.5} |
#+----------------------------+-----------------+-----------------+----------------+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| RandomForestClassifier     | accuracy        |         1647.49 |       0.95939  |      0.910909 | {'bootstrap': False, 'max_depth': 16, 'max_features': 12, 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 100}                                                             |
#+----------------------------+-----------------+-----------------+----------------+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| GradientBoostingClassifier | accuracy        |         1179.07 |       0.918338 |      0.910201 | {'learning_rate': 0.25, 'max_depth': 16, 'max_features': 12, 'min_samples_leaf': 5, 'min_samples_split': 0.1, 'n_estimators': 30}                                                          |
#+----------------------------+-----------------+-----------------+----------------+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| KNeighborsClassifier       | accuracy        |          161.14 |       0.890114 |      0.895249 | {'n_neighbors': 32, 'p': 5}                                                                                                                                                                |
#+----------------------------+-----------------+-----------------+----------------+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| LogisticRegression         | accuracy        |           10.9  |       0.899906 |      0.907724 | {'C': 1}                                                                                                                                                                                   |
#+----------------------------+-----------------+-----------------+----------------+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| DecisionTreeClassifier     | f1              |           29.16 |       0.666196 |      0.471384 | {'max_depth': 12, 'max_features': 38, 'min_samples_leaf': 5, 'min_samples_split': 10}                                                                                                      |
#+----------------------------+-----------------+-----------------+----------------+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| XGBClassifier              | f1              |          751.78 |       0.4952   |      0.421452 | {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 10, 'min_child_weight': 12, 'n_estimators': 20, 'nthread': 4, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.7} |
#+----------------------------+-----------------+-----------------+----------------+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| RandomForestClassifier     | f1              |         1671.34 |       0.792691 |      0.543342 | {'bootstrap': True, 'max_depth': 16, 'max_features': 51, 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 50}                                                               |
#+----------------------------+-----------------+-----------------+----------------+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| GradientBoostingClassifier | f1              |         1206.35 |       0.778455 |      0.542643 | {'learning_rate': 0.25, 'max_depth': 8, 'max_features': 25, 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 30}                                                            |
#+----------------------------+-----------------+-----------------+----------------+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| KNeighborsClassifier       | f1              |          151.45 |       0.38512  |      0.282876 | {'n_neighbors': 8, 'p': 1}                                                                                                                                                                 |
#+----------------------------+-----------------+-----------------+----------------+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| LogisticRegression         | f1              |           10.84 |       0.447771 |      0.467586 | {'C': 1}                                                                                                                                                                                   |
#+----------------------------+-----------------+-----------------+----------------+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| DecisionTreeClassifier     | precision       |           28.62 |       0.653612 |      0.627072 | {'max_depth': 2, 'max_features': 38, 'min_samples_leaf': 5, 'min_samples_split': 10}                                                                                                       |
#+----------------------------+-----------------+-----------------+----------------+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| XGBClassifier              | precision       |          750.57 |       0.781785 |      0.71875  | {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'max_depth': 10, 'min_child_weight': 20, 'n_estimators': 20, 'nthread': 4, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.7} |
#+----------------------------+-----------------+-----------------+----------------+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| RandomForestClassifier     | precision       |         1690.59 |       0.848485 |      0.666667 | {'bootstrap': False, 'max_depth': 2, 'max_features': 12, 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 100}                                                              |
#+----------------------------+-----------------+-----------------+----------------+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| GradientBoostingClassifier | precision       |         1193.19 |       0.822086 |      0.764706 | {'learning_rate': 0.25, 'max_depth': 2, 'max_features': 38, 'min_samples_leaf': 5, 'min_samples_split': 1.0, 'n_estimators': 4}                                                            |
#+----------------------------+-----------------+-----------------+----------------+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| KNeighborsClassifier       | precision       |          161.41 |       0.612491 |      0.596618 | {'n_neighbors': 32, 'p': 5}                                                                                                                                                                |
#+----------------------------+-----------------+-----------------+----------------+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| LogisticRegression         | precision       |           10.52 |       0.654853 |      0.665568 | {'C': 0.01}                                                                                                                                                                                |
#+----------------------------+-----------------+-----------------+----------------+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| DecisionTreeClassifier     | recall          |           27.05 |       0.675528 |      0.486551 | {'max_depth': 16, 'max_features': 25, 'min_samples_leaf': 5, 'min_samples_split': 10}                                                                                                      |
#+----------------------------+-----------------+-----------------+----------------+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| XGBClassifier              | recall          |          759.83 |       0.365217 |      0.307753 | {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 10, 'min_child_weight': 12, 'n_estimators': 20, 'nthread': 4, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.7} |
#+----------------------------+-----------------+-----------------+----------------+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| RandomForestClassifier     | recall          |         1658.37 |       0.768199 |      0.47943  | {'bootstrap': False, 'max_depth': 16, 'max_features': 25, 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 100}                                                             |
#+----------------------------+-----------------+-----------------+----------------+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| GradientBoostingClassifier | recall          |         1190.39 |       0.622857 |      0.502373 | {'learning_rate': 1, 'max_depth': 16, 'max_features': 25, 'min_samples_leaf': 5, 'min_samples_split': 0.1, 'n_estimators': 30}                                                             |
#+----------------------------+-----------------+-----------------+----------------+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| KNeighborsClassifier       | recall          |          155.94 |       0.261366 |      0.205696 | {'n_neighbors': 8, 'p': 5}                                                                                                                                                                 |
#+----------------------------+-----------------+-----------------+----------------+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| LogisticRegression         | recall          |           11.93 |       0.341863 |      0.362342 | {'C': 1}                                                                                                                                                                                   |
#+----------------------------+-----------------+-----------------+----------------+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| DecisionTreeClassifier     | roc_auc         |           28.48 |       0.703954 |      0.700603 | {'max_depth': 16, 'max_features': 51, 'min_samples_leaf': 5, 'min_samples_split': 0.1}                                                                                                     |
#+----------------------------+-----------------+-----------------+----------------+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| XGBClassifier              | roc_auc         |          760.96 |       0.675213 |      0.644264 | {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 10, 'min_child_weight': 12, 'n_estimators': 20, 'nthread': 4, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.7} |
#+----------------------------+-----------------+-----------------+----------------+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| RandomForestClassifier     | roc_auc         |         1650.14 |       0.798659 |      0.698223 | {'bootstrap': True, 'max_depth': 16, 'max_features': 12, 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 100}                                                              |
#+----------------------------+-----------------+-----------------+----------------+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| GradientBoostingClassifier | roc_auc         |         1191.08 |       0.763064 |      0.726684 | {'learning_rate': 0.25, 'max_depth': 16, 'max_features': 25, 'min_samples_leaf': 5, 'min_samples_split': 0.1, 'n_estimators': 30}                                                          |
#+----------------------------+-----------------+-----------------+----------------+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| KNeighborsClassifier       | roc_auc         |          129.29 |       0.585165 |      0.585637 | {'n_neighbors': 32, 'p': 1}                                                                                                                                                                |
#+----------------------------+-----------------+-----------------+----------------+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| LogisticRegression         | roc_auc         |           10.77 |       0.655175 |      0.665564 | {'C': 0.1}                                                                                                                                                                                 |
#+----------------------------+-----------------+-----------------+----------------+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
