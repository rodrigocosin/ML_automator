# ## Lendo arquivo de Configurações


f = open("C:/Users/rcosin/Desktop/FIA/TCC/titanic.conf", "r")
ct = 0
model_param = ''

arq = f.readlines()
for l in arq:
    exec(l)
    ct = ct+1
    if ct == 6:
        ct = 0
        break

for l in arq:
    ct = ct+1
    if ct == 7:
        exec(l)
        ct = 0
        break
        
for l in arq:
    ct = ct+1
    if ct > 9:
        model_param = model_param + str(l)


# ## Funcoes de apoio


global codigo
codigo = ''

def gera_codigo(txt, ant, dep):
    global codigo
    codigo = codigo + '\n'*ant + txt + '\n'*dep
    
def gera_codigo_df(df, showidx, ant, dep):
    global codigo
    
    txt = tabulate(df, headers='keys', tablefmt='grid',showindex = showidx)
    txt = '#' + txt.replace('\n', '\n#')
    
    codigo = codigo + '\n'*ant + txt + '\n'*dep


global lmod_nome
global lmod_score
global lmod_tempo
global lmod_acc_treino
global lmod_acc_teste
global lmod_acc_hiperparam

lmod_nome = []
lmod_score = []
lmod_tempo = []
lmod_acc_treino = []
lmod_acc_teste = []
lmod_acc_hiperparam = []

def executa_modelo(Score, Modelo, parametros):
    
    global lmod_nome
    global lmod_score
    global lmod_tempo
    global lmod_acc_treino
    global lmod_acc_teste
    global lmod_acc_hiperparam

    print("Inicio: " + Modelo)

    DataA = datetime.datetime.now()
    
    gera_codigo("# Modelo: " + Modelo, 2, 1)

    cmd = Modelo +'()'
    modelo = eval(cmd)
    gera_codigo("modelo = " + cmd, 0, 1)

    gera_codigo("# Tunning: ", 1, 1)

    gera_codigo("parametros = " + str(parametros), 0, 1)

    mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4,  scoring=Score, verbose=0, refit=True)
    gera_codigo("mod = GridSearchCV(modelo, parametros, n_jobs=4,  cv=4, scoring='"+ Score +"', verbose=10, refit=True)", 0, 1)

    mod.fit(X_train, Y_train)
    gera_codigo("mod.fit(X_train, Y_train)", 0, 1)

    gera_codigo("# Melhores Parametros: " + str(mod.best_params_), 1, 1)

    gera_codigo("# Aplicando o Modelo", 1, 1)
    
    cmd = Modelo + '(**mod.best_params_)'
    model = eval(Modelo + '(**mod.best_params_)')
    gera_codigo("model = " + cmd, 0, 1)

    model.fit(X_train, Y_train)
    gera_codigo("model.fit(X_train, Y_train)", 0, 1)

    predict_train = model.predict(X_train)
    gera_codigo("predict_train = model.predict(X_train)", 0, 1)
    
    predict_test = model.predict(X_test)
    gera_codigo("predict_test = model.predict(X_test)", 0, 1)

    DataX = datetime.datetime.now() -DataA
        
    lmod_nome.append(Modelo)
    lmod_score.append(Score)
    lmod_acc_hiperparam.append(mod.best_params_)
    lmod_tempo.append(round(DataX.total_seconds(),2))
    
    gera_codigo("# Calculando o Score", 1, 1)
    
    if Score == 'accuracy':
        lmod_acc_treino.append(round(accuracy_score(Y_train, predict_train),6))
        lmod_acc_teste.append(round(accuracy_score(Y_test, predict_test),6))     
        gera_codigo("score_treino = round(accuracy_score(Y_train, predict_train),6)", 0, 1)
        gera_codigo("score_teste = round(accuracy_score(Y_test, predict_test),6)", 0, 1)
    elif Score == 'f1':
        lmod_acc_treino.append(round(f1_score(Y_train, predict_train),6))
        lmod_acc_teste.append(round(f1_score(Y_test, predict_test),6))
        gera_codigo("score_treino = round(f1_score(Y_train, predict_train),6)", 0, 1)
        gera_codigo("score_teste = round(f1_score(Y_test, predict_test),6)", 0, 1)
    if Score == 'precision':
        lmod_acc_treino.append(round(precision_score(Y_train, predict_train),6))
        lmod_acc_teste.append(round(precision_score(Y_test, predict_test),6))  
        gera_codigo("score_treino = round(precision_score(Y_train, predict_train),6)", 0, 1)
        gera_codigo("score_teste = round(precision_score(Y_test, predict_test),6)", 0, 1)
    if Score == 'recall':
        lmod_acc_treino.append(round(recall_score(Y_train, predict_train),6))
        lmod_acc_teste.append(round(recall_score(Y_test, predict_test),6))  
        gera_codigo("score_treino = round(recall_score(Y_train, predict_train),6)", 0, 1)
        gera_codigo("score_teste = round(recall_score(Y_test, predict_test),6)", 0, 1)
    if Score == 'roc_auc':
        lmod_acc_treino.append(round(roc_auc_score(Y_train, predict_train),6))
        lmod_acc_teste.append(round(roc_auc_score(Y_test, predict_test),6))  
        gera_codigo("score_treino = round(roc_auc_score(Y_train, predict_train),6)", 0, 1)
        gera_codigo("score_teste = round(roc_auc_score(Y_test, predict_test),6)", 0, 1)

    gera_codigo("# ----------------------------------------------", 1, 1)

    print("Finalizado após: " + str(round(DataX.total_seconds(),2)) + " segundos")
    print("--------------------")


# ## Importação de Modulos e Pacotes


import warnings
warnings.filterwarnings('ignore')

gera_codigo("# Importando Modulos e Pacotes", 0, 1)
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

gera_codigo("import numpy as np", 0, 1)
gera_codigo("import pandas as pd", 0, 1)
gera_codigo("from tabulate import tabulate", 0, 1)
gera_codigo("import sklearn.model_selection as model_selection", 0, 1)
gera_codigo("from sklearn.tree import DecisionTreeClassifier", 0, 1)
gera_codigo("from sklearn.linear_model import LogisticRegression", 0, 1)
gera_codigo("from sklearn.neighbors import KNeighborsClassifier", 0, 1)
gera_codigo("from sklearn.naive_bayes import GaussianNB", 0, 1)
gera_codigo("from sklearn.svm import SVC, LinearSVC", 0, 1)
gera_codigo("from sklearn.ensemble import RandomForestClassifier", 0, 1)
gera_codigo("from sklearn.ensemble import GradientBoostingClassifier", 0, 1)
gera_codigo("from sklearn.metrics import accuracy_score", 0, 1)
gera_codigo("from sklearn.metrics import roc_auc_score", 0, 1)
gera_codigo("from sklearn.metrics import f1_score", 0, 1)
gera_codigo("from sklearn.metrics import precision_score", 0, 1)
gera_codigo("from sklearn.metrics import recall_score", 0, 1)
gera_codigo("from sklearn.metrics import auc", 0, 1)
gera_codigo("from sklearn.model_selection import GridSearchCV", 0, 1)
gera_codigo("from xgboost import XGBClassifier", 0, 1)
gera_codigo("import datetime", 0, 2)


# ## Criando Data Frame inicial


gera_codigo("# Atribuindo parametros de entrada", 0, 1)

gera_codigo("target = '" + target +"'", 0, 2)

gera_codigo("# Importando os dados de entrada", 0, 1)

df = pd.read_csv(dados, sep = delimitador, decimal = separador_decimal)
gera_codigo("df = pd.read_csv('"+ dados +"', sep = '"+ delimitador \
            +"', decimal = '"+ separador_decimal +"')", 0, 2)


# ## Seleção de Variáveis e Identificação de tratamento


# Criacao de variaveis que serao utilizadas no Dataframe
# responsavel por consolidar todos os dados para
# selecao de variaveis
lcol = []
ltip = []
ldesvio = []
lmedia = []
lcoefvar = []

lmediana = []
lcoecentral = []

ldistper = []
ldistval = []
lnull = []

lacao = []
ldescacao = []

for x in df.columns :
    
    if x != target:

        lcol.append(x)

        cmd = 'df.' + x + '.dtype'
        ltip.append(str(eval(cmd)))

        if str(eval(cmd)) in ('float64','int64'):
            cmd = 'round( df.' + x + '.std() , 2)'
            ldesvio.append(str(eval(cmd)))
            
            cmd = 'round( df.' + x + '.mean() , 2)'
            lmedia.append(str(eval(cmd)))
            
            if abs(float(lmedia[-1])) == 0:
                cv = 0
            else:
                cv = round((float(ldesvio[-1]) / float(lmedia[-1])) * 100 ,2)
                
            lcoefvar.append(str(cv))
            
            cmd = 'round( df.' + x + '.median() , 2)'
            lmediana.append(str(eval(cmd)))

            if abs(float(lmedia[-1])) == 0:
                central = 0
            else:
                central = round((float(lmediana[-1]) / float(lmedia[-1])) * 100 ,2)
                
            lcoecentral.append(str(central))
            
        else:
            ldesvio.append('-')
            lmedia.append('-')
            lcoefvar.append('-')
            lmediana.append('-')
            lcoecentral.append('-')

        cmd = 'round( df.' + x + '.value_counts().count() / len(df.index) * 100 ,2)'
        ldistper.append(str(eval(cmd)))
        
        cmd = 'df.' + x + '.value_counts().count()'
        ldistval.append(str(eval(cmd)))

        cmd = 'round( df.' + x + '.isna().sum() / len(df.index) * 100 ,2)'
        lnull.append(str(eval(cmd)))
        
        if ltip[-1] in ('float64','int64') and float(ldistper[-1]) > 70 and \
            float(lcoefvar[-1]) > 50 and float(lcoecentral[-1]) > 80:
            lacao.append('excluir')
            ldescacao.append('alta dispersao')
            
        elif float(lnull[-1]) > 75:
            lacao.append('excluir')
            ldescacao.append('muito nulo')
            
        elif float(ldistval[-1]) < 30 and float(lnull[-1]) > 0 :
            lacao.append('dummy_com_null')
            ldescacao.append('-')
            
        elif float(ldistval[-1]) < 30 and float(lnull[-1]) == 0 :
            lacao.append('dummy_sem_null')
            ldescacao.append('-')
            
        elif ltip[-1] in ('float64','int64') and float(lnull[-1]) > 0 :
            lacao.append('continua_com_null')
            ldescacao.append('-')

        elif ltip[-1] in ('float64','int64') and float(lnull[-1]) == 0 :
            lacao.append('continua_sem_null')
            ldescacao.append('-')
            
        elif ltip[-1] in ('object') and float(ldistper[-1]) > 90:
            lacao.append('excluir')
            ldescacao.append('categorica com muitas categorias')
            
        elif ltip[-1] in ('object') :
            lacao.append('filtrar')
            ldescacao.append('possivel Feature Engineering')
            
        else:
            lacao.append('nao especificado')
            ldescacao.append('-')
    
lfinal =  list(zip(lcol, ltip, ldesvio, lmedia, lcoefvar, lmediana, lcoecentral, \
                   ldistper, ldistval, lnull, lacao, ldescacao))

df_Sel_Variaveis = pd.DataFrame(lfinal, columns = ['Coluna' , 'Tipo', 'Desvio Padrao', \
                                                   'Media', 'Coef de Variacao', \
                                                   'Mediana',  'Coef de Centralidade', \
                                                   'Perc_Distintos', 'Valores_Distintos', \
                                                   'Perc_Null', 'Acao', 'Desc Acao']) 

gera_codigo("# Analise de variaveis e identificacao de tratamento:", 0, 1)
gera_codigo_df(df_Sel_Variaveis,False,0,2)

gera_codigo("# Correlacao entre as variaveis:", 0, 1)
gera_codigo_df(df.corr().round(3),True,0,2)


# ## Tratamento de Dados


# Criacao de variaveis que serao utilizadas no Dataframe
# responsavel por consolidar todos os dados para
# selecao de variaveis
lcol = []
ltip = []
ldesvio = []
lmedia = []
lcoefvar = []

lmediana = []
lcoecentral = []

ldistper = []
ldistval = []
lnull = []

lacao = []
ldescacao = []

for x in df.columns :
    
    if x != target:

        lcol.append(x)

        cmd = 'df.' + x + '.dtype'
        ltip.append(str(eval(cmd)))

        if str(eval(cmd)) in ('float64','int64'):
            cmd = 'round( df.' + x + '.std() , 2)'
            ldesvio.append(str(eval(cmd)))
            
            cmd = 'round( df.' + x + '.mean() , 2)'
            lmedia.append(str(eval(cmd)))
            
            if abs(float(lmedia[-1])) == 0:
                cv = 0
            else:
                cv = round((float(ldesvio[-1]) / float(lmedia[-1])) * 100 ,2)
                
            lcoefvar.append(str(cv))
            
            cmd = 'round( df.' + x + '.median() , 2)'
            lmediana.append(str(eval(cmd)))

            if abs(float(lmedia[-1])) == 0:
                central = 0
            else:
                central = round((float(lmediana[-1]) / float(lmedia[-1])) * 100 ,2)
                
            lcoecentral.append(str(central))
            
        else:
            ldesvio.append('-')
            lmedia.append('-')
            lcoefvar.append('-')
            lmediana.append('-')
            lcoecentral.append('-')

        cmd = 'round( df.' + x + '.value_counts().count() / len(df.index) * 100 ,2)'
        ldistper.append(str(eval(cmd)))
        
        cmd = 'df.' + x + '.value_counts().count()'
        ldistval.append(str(eval(cmd)))

        cmd = 'round( df.' + x + '.isna().sum() / len(df.index) * 100 ,2)'
        lnull.append(str(eval(cmd)))
        
        if ltip[-1] in ('float64','int64') and float(ldistper[-1]) > 70 and \
            float(lcoefvar[-1]) > 50 and float(lcoecentral[-1]) > 80:
            lacao.append('excluir')
            ldescacao.append('alta dispersao')
            
        elif float(lnull[-1]) > 75:
            lacao.append('excluir')
            ldescacao.append('muito nulo')
            
        elif float(ldistval[-1]) < 30 and float(lnull[-1]) > 0 :
            lacao.append('dummy_com_null')
            ldescacao.append('-')
            
        elif float(ldistval[-1]) < 30 and float(lnull[-1]) == 0 :
            lacao.append('dummy_sem_null')
            ldescacao.append('-')
            
        elif ltip[-1] in ('float64','int64') and float(lnull[-1]) > 0 :
            lacao.append('continua_com_null')
            ldescacao.append('-')

        elif ltip[-1] in ('float64','int64') and float(lnull[-1]) == 0 :
            lacao.append('continua_sem_null')
            ldescacao.append('-')
            
        elif ltip[-1] in ('object') and float(ldistper[-1]) > 90:
            lacao.append('excluir')
            ldescacao.append('categorica com muitas categorias')
            
        elif ltip[-1] in ('object') :
            lacao.append('filtrar')
            ldescacao.append('possivel Feature Engineering')
            
        else:
            lacao.append('nao especificado')
            ldescacao.append('-')
    
lfinal =  list(zip(lcol, ltip, ldesvio, lmedia, lcoefvar, lmediana, lcoecentral, \
                   ldistper, ldistval, lnull, lacao, ldescacao))

df_Sel_Variaveis = pd.DataFrame(lfinal, columns = ['Coluna' , 'Tipo', 'Desvio Padrao', \
                                                   'Media', 'Coef de Variacao', \
                                                   'Mediana',  'Coef de Centralidade', \
                                                   'Perc_Distintos', 'Valores_Distintos', \
                                                   'Perc_Null', 'Acao', 'Desc Acao']) 

gera_codigo("# Analise de variaveis e identificacao de tratamento:", 0, 1)
gera_codigo_df(df_Sel_Variaveis,False,0,2)

gera_codigo("# Correlacao entre as variaveis:", 0, 1)
gera_codigo_df(df.corr().round(3),True,0,2)


# ## Separando bases de teste e treino


gera_codigo("# Dividindo a base entre variaveis explicativas e variavel resposta", 2, 1)
X = df2.loc[:, df2.columns != target]
Y = eval("df2." + target)
gera_codigo("X = df2.loc[:, df2.columns !='"+ target +"']", 0, 1)
gera_codigo("Y = df2."+ target , 0, 1)

gera_codigo("# Separando a Base entre Teste e Treino", 1, 1)
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=0.75, test_size=0.25, random_state=7)
gera_codigo("X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=0.75, test_size=0.25, random_state=7)", 0, 1)



# ## Aplicando e Ajustando Modelos


for s in score:

    print('--------\nScore: ' + s + '\n--------')
    
    for m in model_param.split('\n'):
    

        mod = str(m.split(';')[0])
        par = eval(m.split(';')[1])
    
        executa_modelo(s, mod , par)


# ## Comparativo de Modelos


gera_codigo("# Comparativo de Modelos: ",1,1)

l_modelos =  list(zip(lmod_nome, lmod_score, lmod_tempo, lmod_acc_treino, lmod_acc_teste,\
                      lmod_acc_hiperparam))
df_modelos = pd.DataFrame(l_modelos, columns = ['Modelo' , 'Metrica_Score', 'Tempo_Exec(s)', 'Score_Treino',\
                                                'Score_Teste', 'Melhor_hiper_param']) 

gera_codigo_df(df_modelos,False,1,1)


# ## Gerando codigo.py


text_file = open(codigo_out, "w")
text_file.write(codigo)
text_file.close()
