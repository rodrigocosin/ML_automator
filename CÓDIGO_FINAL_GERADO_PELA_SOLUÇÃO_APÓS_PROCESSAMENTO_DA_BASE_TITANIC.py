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
target = 'Survived'

# Importando os dados de entrada
df = pd.read_csv('C:/Users/rcosin/Desktop/FIA/TCC/Titanic/train.csv', sep = ',', decimal = '.')

# Analise de variaveis e identificacao de tratamento:
#+-------------+---------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+----------------------------------+
#| Coluna      | Tipo    | Desvio Padrao   | Media   | Coef de Variacao   | Mediana   | Coef de Centralidade   |   Perc_Distintos |   Valores_Distintos |   Perc_Null | Acao              | Desc Acao                        |
#+=============+=========+=================+=========+====================+===========+========================+==================+=====================+=============+===================+==================================+
#| PassengerId | int64   | 257.35          | 446.0   | 57.7               | 446.0     | 100.0                  |           100    |                 891 |        0    | excluir           | alta dispersao                   |
#+-------------+---------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+----------------------------------+
#| Pclass      | int64   | 0.84            | 2.31    | 36.36              | 3.0       | 129.87                 |             0.34 |                   3 |        0    | dummy_sem_null    | -                                |
#+-------------+---------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+----------------------------------+
#| Name        | object  | -               | -       | -                  | -         | -                      |           100    |                 891 |        0    | excluir           | categorica com muitas categorias |
#+-------------+---------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+----------------------------------+
#| Sex         | object  | -               | -       | -                  | -         | -                      |             0.22 |                   2 |        0    | dummy_sem_null    | -                                |
#+-------------+---------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+----------------------------------+
#| Age         | float64 | 14.53           | 29.7    | 48.92              | 28.0      | 94.28                  |             9.88 |                  88 |       19.87 | continua_com_null | -                                |
#+-------------+---------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+----------------------------------+
#| SibSp       | int64   | 1.1             | 0.52    | 211.54             | 0.0       | 0.0                    |             0.79 |                   7 |        0    | dummy_sem_null    | -                                |
#+-------------+---------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+----------------------------------+
#| Parch       | int64   | 0.81            | 0.38    | 213.16             | 0.0       | 0.0                    |             0.79 |                   7 |        0    | dummy_sem_null    | -                                |
#+-------------+---------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+----------------------------------+
#| Ticket      | object  | -               | -       | -                  | -         | -                      |            76.43 |                 681 |        0    | filtrar           | possivel Feature Engineering     |
#+-------------+---------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+----------------------------------+
#| Fare        | float64 | 49.69           | 32.2    | 154.32             | 14.45     | 44.88                  |            27.83 |                 248 |        0    | continua_sem_null | -                                |
#+-------------+---------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+----------------------------------+
#| Cabin       | object  | -               | -       | -                  | -         | -                      |            16.5  |                 147 |       77.1  | excluir           | muito nulo                       |
#+-------------+---------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+----------------------------------+
#| Embarked    | object  | -               | -       | -                  | -         | -                      |             0.34 |                   3 |        0.22 | dummy_com_null    | -                                |
#+-------------+---------+-----------------+---------+--------------------+-----------+------------------------+------------------+---------------------+-------------+-------------------+----------------------------------+

# Correlacao entre as variaveis:
#+-------------+---------------+------------+----------+--------+---------+---------+--------+
#|             |   PassengerId |   Survived |   Pclass |    Age |   SibSp |   Parch |   Fare |
#+=============+===============+============+==========+========+=========+=========+========+
#| PassengerId |         1     |     -0.005 |   -0.035 |  0.037 |  -0.058 |  -0.002 |  0.013 |
#+-------------+---------------+------------+----------+--------+---------+---------+--------+
#| Survived    |        -0.005 |      1     |   -0.338 | -0.077 |  -0.035 |   0.082 |  0.257 |
#+-------------+---------------+------------+----------+--------+---------+---------+--------+
#| Pclass      |        -0.035 |     -0.338 |    1     | -0.369 |   0.083 |   0.018 | -0.549 |
#+-------------+---------------+------------+----------+--------+---------+---------+--------+
#| Age         |         0.037 |     -0.077 |   -0.369 |  1     |  -0.308 |  -0.189 |  0.096 |
#+-------------+---------------+------------+----------+--------+---------+---------+--------+
#| SibSp       |        -0.058 |     -0.035 |    0.083 | -0.308 |   1     |   0.415 |  0.16  |
#+-------------+---------------+------------+----------+--------+---------+---------+--------+
#| Parch       |        -0.002 |      0.082 |    0.018 | -0.189 |   0.415 |   1     |  0.216 |
#+-------------+---------------+------------+----------+--------+---------+---------+--------+
#| Fare        |         0.013 |      0.257 |   -0.549 |  0.096 |   0.16  |   0.216 |  1     |
#+-------------+---------------+------------+----------+--------+---------+---------+--------+


# Tratamento de Dados
df2 = pd.concat([df['Survived']])

# Criando Dataframe auxiliar
df_temp = pd.DataFrame()

# Adicionando colunas continuas sem nulos
df2 = pd.concat([df['Fare'], df2], axis=1)

# Adicionando colunas continuas com nulos
df_temp = df.Age.fillna( df.Age.mean() )
df2 = pd.concat([df_temp, df2], axis=1)

# Adicionando e preparando colunas com dummies sem tratamento de NULL
df_temp = pd.get_dummies( df.Pclass, prefix='Pclass' )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = pd.get_dummies( df.Sex, prefix='Sex' )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = pd.get_dummies( df.SibSp, prefix='SibSp' )
df2 = pd.concat([df_temp, df2], axis=1)
df_temp = pd.get_dummies( df.Parch, prefix='Parch' )
df2 = pd.concat([df_temp, df2], axis=1)

# Adicionando e preparando colunas com dummies com tratamento de NULL
df_temp = df.Embarked.fillna( 'NULL' )
df_temp = pd.get_dummies( df_temp, prefix='Embarked' )
df2 = pd.concat([df_temp, df2], axis=1)


# Dividindo a base entre variaveis explicativas e variavel resposta
X = df2.loc[:, df2.columns !='Survived']
Y = df2.Survived

# Separando a Base entre Teste e Treino
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=0.75, test_size=0.25, random_state=7)


# Modelo: DecisionTreeClassifier
modelo = DecisionTreeClassifier()

# Tunning: 
parametros = {'max_depth': [2, 4, 8, 12, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'max_features': [6, 12, 18, 25]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='accuracy', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'max_depth': 12, 'max_features': 18, 'min_samples_leaf': 5, 'min_samples_split': 0.1}

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

# Melhores Parametros: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'max_depth': 8, 'min_child_weight': 12, 'n_estimators': 10, 'nthread': 4, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.6}

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
parametros = {'n_estimators': [10, 50, 100], 'max_depth': [2, 8, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'bootstrap': [True, False], 'max_features': [6, 12, 18, 25]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='accuracy', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'bootstrap': False, 'max_depth': 16, 'max_features': 12, 'min_samples_leaf': 5, 'min_samples_split': 0.1, 'n_estimators': 10}

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
parametros = {'learning_rate': [1, 0.25, 0.05, 0.01], 'n_estimators': [4, 16, 100], 'max_depth': [2, 8, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'max_features': [6, 12, 18, 25]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='accuracy', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'learning_rate': 0.25, 'max_depth': 8, 'max_features': 18, 'min_samples_leaf': 0.1, 'min_samples_split': 10, 'n_estimators': 100}

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

# Melhores Parametros: {'n_neighbors': 8, 'p': 1}

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

# Melhores Parametros: {'C': 0.1}

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

# Melhores Parametros: {'C': 10, 'degree': 0, 'gamma': 0.05}

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
parametros = {'max_depth': [2, 4, 8, 12, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'max_features': [6, 12, 18, 25]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='f1', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'max_depth': 8, 'max_features': 18, 'min_samples_leaf': 5, 'min_samples_split': 0.1}

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

# Melhores Parametros: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'max_depth': 8, 'min_child_weight': 12, 'n_estimators': 10, 'nthread': 4, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.6}

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
parametros = {'n_estimators': [10, 50, 100], 'max_depth': [2, 8, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'bootstrap': [True, False], 'max_features': [6, 12, 18, 25]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='f1', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'bootstrap': True, 'max_depth': 8, 'max_features': 18, 'min_samples_leaf': 5, 'min_samples_split': 0.1, 'n_estimators': 10}

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
parametros = {'learning_rate': [1, 0.25, 0.05, 0.01], 'n_estimators': [4, 16, 100], 'max_depth': [2, 8, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'max_features': [6, 12, 18, 25]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='f1', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'learning_rate': 0.05, 'max_depth': 16, 'max_features': 6, 'min_samples_leaf': 5, 'min_samples_split': 0.1, 'n_estimators': 100}

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

# Melhores Parametros: {'C': 0.1}

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
parametros = {'max_depth': [2, 4, 8, 12, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'max_features': [6, 12, 18, 25]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='precision', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'max_depth': 4, 'max_features': 12, 'min_samples_leaf': 0.1, 'min_samples_split': 0.1}

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

# Melhores Parametros: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'max_depth': 8, 'min_child_weight': 12, 'n_estimators': 10, 'nthread': 4, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.6}

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
parametros = {'n_estimators': [10, 50, 100], 'max_depth': [2, 8, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'bootstrap': [True, False], 'max_features': [6, 12, 18, 25]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='precision', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'bootstrap': False, 'max_depth': 16, 'max_features': 6, 'min_samples_leaf': 5, 'min_samples_split': 1.0, 'n_estimators': 10}

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
parametros = {'learning_rate': [1, 0.25, 0.05, 0.01], 'n_estimators': [4, 16, 100], 'max_depth': [2, 8, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'max_features': [6, 12, 18, 25]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='precision', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'learning_rate': 0.05, 'max_depth': 2, 'max_features': 25, 'min_samples_leaf': 0.1, 'min_samples_split': 0.1, 'n_estimators': 16}

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

# Melhores Parametros: {'n_neighbors': 8, 'p': 1}

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


# Modelo: SVC
modelo = SVC()

# Tunning: 
parametros = {'gamma': [0.05, 0.1, 1, 5, 10], 'C': [0.1, 1, 10, 100, 1000], 'degree': [0, 1, 2, 3, 5, 7]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='precision', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'C': 1, 'degree': 0, 'gamma': 5}

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
parametros = {'max_depth': [2, 4, 8, 12, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'max_features': [6, 12, 18, 25]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='recall', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'max_depth': 4, 'max_features': 6, 'min_samples_leaf': 0.1, 'min_samples_split': 0.1}

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

# Melhores Parametros: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'max_depth': 8, 'min_child_weight': 12, 'n_estimators': 5, 'nthread': 4, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.5}

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
parametros = {'n_estimators': [10, 50, 100], 'max_depth': [2, 8, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'bootstrap': [True, False], 'max_features': [6, 12, 18, 25]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='recall', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'bootstrap': False, 'max_depth': 8, 'max_features': 25, 'min_samples_leaf': 5, 'min_samples_split': 0.1, 'n_estimators': 10}

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
parametros = {'learning_rate': [1, 0.25, 0.05, 0.01], 'n_estimators': [4, 16, 100], 'max_depth': [2, 8, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'max_features': [6, 12, 18, 25]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='recall', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'learning_rate': 0.25, 'max_depth': 16, 'max_features': 18, 'min_samples_leaf': 5, 'min_samples_split': 0.1, 'n_estimators': 16}

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

# Melhores Parametros: {'n_neighbors': 8, 'p': 1}

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

# Melhores Parametros: {'C': 0.1}

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
parametros = {'max_depth': [2, 4, 8, 12, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'max_features': [6, 12, 18, 25]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='roc_auc', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'max_depth': 12, 'max_features': 25, 'min_samples_leaf': 5, 'min_samples_split': 0.1}

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

# Melhores Parametros: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'max_depth': 8, 'min_child_weight': 12, 'n_estimators': 20, 'nthread': 4, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.7}

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
parametros = {'n_estimators': [10, 50, 100], 'max_depth': [2, 8, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'bootstrap': [True, False], 'max_features': [6, 12, 18, 25]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='roc_auc', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'bootstrap': False, 'max_depth': 16, 'max_features': 6, 'min_samples_leaf': 5, 'min_samples_split': 0.1, 'n_estimators': 10}

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
parametros = {'learning_rate': [1, 0.25, 0.05, 0.01], 'n_estimators': [4, 16, 100], 'max_depth': [2, 8, 16], 'min_samples_split': [0.1, 1.0, 10], 'min_samples_leaf': [0.1, 0.5, 5], 'max_features': [6, 12, 18, 25]}
mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='roc_auc', verbose=10, refit=True)
mod.fit(X_train, Y_train)

# Melhores Parametros: {'learning_rate': 0.05, 'max_depth': 2, 'max_features': 18, 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 100}

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

# Melhores Parametros: {'n_neighbors': 8, 'p': 1}

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

# Melhores Parametros: {'C': 1}

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

# Melhores Parametros: {'C': 10, 'degree': 0, 'gamma': 0.05}

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
#| DecisionTreeClassifier     | accuracy        |            9.49 |       0.845808 |      0.753363 | {'max_depth': 12, 'max_features': 18, 'min_samples_leaf': 5, 'min_samples_split': 0.1}                                                                                                    |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| XGBClassifier              | accuracy        |            7.38 |       0.80988  |      0.753363 | {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'max_depth': 8, 'min_child_weight': 12, 'n_estimators': 10, 'nthread': 4, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.6} |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| RandomForestClassifier     | accuracy        |           84.66 |       0.844311 |      0.802691 | {'bootstrap': True, 'max_depth': 16, 'max_features': 12, 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 10}                                                              |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| GradientBoostingClassifier | accuracy        |           78.82 |       0.898204 |      0.789238 | {'learning_rate': 0.25, 'max_depth': 8, 'max_features': 18, 'min_samples_leaf': 0.1, 'min_samples_split': 10, 'n_estimators': 100}                                                        |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| KNeighborsClassifier       | accuracy        |            1.09 |       0.782934 |      0.721973 | {'n_neighbors': 8, 'p': 1}                                                                                                                                                                |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| LogisticRegression         | accuracy        |            0.3  |       0.823353 |      0.748879 | {'C': 0.1}                                                                                                                                                                                |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| SVC                        | accuracy        |           13.23 |       0.932635 |      0.690583 | {'C': 10, 'degree': 0, 'gamma': 0.05}                                                                                                                                                     |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| DecisionTreeClassifier     | f1              |            2.17 |       0.785863 |      0.654088 | {'max_depth': 8, 'max_features': 18, 'min_samples_leaf': 5, 'min_samples_split': 0.1}                                                                                                     |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| XGBClassifier              | f1              |            9.26 |       0.728051 |      0.645161 | {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'max_depth': 8, 'min_child_weight': 12, 'n_estimators': 10, 'nthread': 4, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.6} |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| RandomForestClassifier     | f1              |           79.82 |       0.779874 |      0.675    | {'bootstrap': True, 'max_depth': 8, 'max_features': 18, 'min_samples_leaf': 5, 'min_samples_split': 0.1, 'n_estimators': 10}                                                              |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| GradientBoostingClassifier | f1              |           80.08 |       0.852391 |      0.696203 | {'learning_rate': 0.05, 'max_depth': 16, 'max_features': 6, 'min_samples_leaf': 5, 'min_samples_split': 0.1, 'n_estimators': 100}                                                         |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| KNeighborsClassifier       | f1              |            1.07 |       0.662005 |      0.550725 | {'n_neighbors': 8, 'p': 1}                                                                                                                                                                |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| LogisticRegression         | f1              |            0.21 |       0.757202 |      0.658537 | {'C': 0.1}                                                                                                                                                                                |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| SVC                        | f1              |           14.47 |       0.909457 |      0.596491 | {'C': 10, 'degree': 0, 'gamma': 0.05}                                                                                                                                                     |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| DecisionTreeClassifier     | precision       |            2.18 |       0.790055 |      0.775862 | {'max_depth': 4, 'max_features': 12, 'min_samples_leaf': 0.1, 'min_samples_split': 0.1}                                                                                                   |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| XGBClassifier              | precision       |            8.17 |       0.798122 |      0.746269 | {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'max_depth': 8, 'min_child_weight': 12, 'n_estimators': 10, 'nthread': 4, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.6} |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| RandomForestClassifier     | precision       |           77.9  |       0.975    |      0.933333 | {'bootstrap': False, 'max_depth': 16, 'max_features': 6, 'min_samples_leaf': 5, 'min_samples_split': 1.0, 'n_estimators': 10}                                                             |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| GradientBoostingClassifier | precision       |           86.42 |       0.943548 |      0.956522 | {'learning_rate': 0.05, 'max_depth': 2, 'max_features': 25, 'min_samples_leaf': 0.1, 'min_samples_split': 0.1, 'n_estimators': 16}                                                        |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| KNeighborsClassifier       | precision       |            1.04 |       0.811429 |      0.76     | {'n_neighbors': 8, 'p': 1}                                                                                                                                                                |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| LogisticRegression         | precision       |            0.17 |       0.801325 |      0.877551 | {'C': 0.01}                                                                                                                                                                               |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| SVC                        | precision       |           13.68 |       0.995868 |      0.411765 | {'C': 1, 'degree': 0, 'gamma': 5}                                                                                                                                                         |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| DecisionTreeClassifier     | recall          |            2.45 |       0.255906 |      0.272727 | {'max_depth': 4, 'max_features': 6, 'min_samples_leaf': 0.1, 'min_samples_split': 0.1}                                                                                                    |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| XGBClassifier              | recall          |            8.15 |       0.606299 |      0.545455 | {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'max_depth': 8, 'min_child_weight': 12, 'n_estimators': 5, 'nthread': 4, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.5}  |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| RandomForestClassifier     | recall          |           78.14 |       0.748031 |      0.613636 | {'bootstrap': False, 'max_depth': 8, 'max_features': 25, 'min_samples_leaf': 5, 'min_samples_split': 0.1, 'n_estimators': 10}                                                             |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| GradientBoostingClassifier | recall          |           83.62 |       0.866142 |      0.647727 | {'learning_rate': 0.25, 'max_depth': 16, 'max_features': 18, 'min_samples_leaf': 5, 'min_samples_split': 0.1, 'n_estimators': 16}                                                         |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| KNeighborsClassifier       | recall          |            1.41 |       0.559055 |      0.431818 | {'n_neighbors': 8, 'p': 1}                                                                                                                                                                |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| LogisticRegression         | recall          |            0.25 |       0.724409 |      0.613636 | {'C': 0.1}                                                                                                                                                                                |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| SVC                        | recall          |           13.2  |       0.889764 |      0.579545 | {'C': 10, 'degree': 0, 'gamma': 0.05}                                                                                                                                                     |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| DecisionTreeClassifier     | roc_auc         |            2.64 |       0.826914 |      0.72904  | {'max_depth': 12, 'max_features': 25, 'min_samples_leaf': 5, 'min_samples_split': 0.1}                                                                                                    |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| XGBClassifier              | roc_auc         |            8.18 |       0.787459 |      0.706566 | {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'max_depth': 8, 'min_child_weight': 12, 'n_estimators': 20, 'nthread': 4, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.7} |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| RandomForestClassifier     | roc_auc         |           78.42 |       0.797834 |      0.736195 | {'bootstrap': False, 'max_depth': 16, 'max_features': 6, 'min_samples_leaf': 5, 'min_samples_split': 0.1, 'n_estimators': 10}                                                             |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| GradientBoostingClassifier | roc_auc         |           83.74 |       0.830984 |      0.73447  | {'learning_rate': 0.05, 'max_depth': 2, 'max_features': 18, 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 100}                                                          |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| KNeighborsClassifier       | roc_auc         |            0.68 |       0.739672 |      0.671465 | {'n_neighbors': 8, 'p': 1}                                                                                                                                                                |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| LogisticRegression         | roc_auc         |            0.18 |       0.818907 |      0.740404 | {'C': 1}                                                                                                                                                                                  |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
#| SVC                        | roc_auc         |           13.28 |       0.92435  |      0.671254 | {'C': 10, 'degree': 0, 'gamma': 0.05}                                                                                                                                                     |
#+----------------------------+-----------------+-----------------+----------------+---------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
