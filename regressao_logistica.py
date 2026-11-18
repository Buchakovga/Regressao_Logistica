
# %%

#carregando pacotes
import pandas as pd
import numpy as np
import seaborn as sns
import sys
import statistics
import math
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import sklearn
from sklearn.metrics import roc_curve, auc


# %%

#assim como na regressão linear, criamos funções personalizadas para os métodos de seleção de variáveis
#carregando funções, todas de uma vez
from funcoes_importantes_regressao_logistica import step,univariada_variavel_numerica,univariada_variavel_categoria,analise_var_numerica_por_percentil


# %%

#importando base bank
bank = pd.read_csv("bank.csv",sep=";")
bank


# %%

# observando missing nas variáveis
bank.isnull().sum()


# %%


#Analise univariada para variáveis numéricas
univariada_variavel_numerica(bank, "age")
# %%

univariada_variavel_numerica(bank, "balance")

# %%

univariada_variavel_numerica(bank, "duration")

# %%

univariada_variavel_numerica(bank, "campaign")

# %%


univariada_variavel_numerica(bank, "pdays")

# %%

univariada_variavel_numerica(bank, "previous")




# %%

univariada_variavel_categoria(bank, "job")

# %%

univariada_variavel_categoria(bank, "marital")

# %%


univariada_variavel_categoria(bank, "education")


#%%

univariada_variavel_categoria(bank, "default")

#%%
univariada_variavel_categoria(bank, "housing")

#%%

univariada_variavel_categoria(bank, "loan")

# %%

univariada_variavel_categoria(bank, "contact")

# %%


univariada_variavel_categoria(bank, "poutcome")


# %%



# %%
univariada_variavel_categoria(bank, "y")

# %%


#######
######     Análise Bivariada das Variáveis
#####


#############################################
#    IDADE
#############################################

# criando uma copia para manipular o que importa
df_age_cm = bank[['y', 'age']].copy()
# transformando em 1 e 0 

# Mapeia explicitamente 'yes' para 1 e 'no' para 0
df_age_cm['y'] = df_age_cm['y'].map({'yes': 1, 'no': 0})

# ordenando pela Idade
df_age_cm = df_age_cm.sort_values("age").reset_index()
# criando uma variavel com o n_total da base
n_total = df_age_cm['y'].count()
# criando a variavel percentil da idade, de 10 em 10 %
df_age_cm['percentil_idade']=pd.cut(
   df_age_cm.index,
   bins=[-1, (n_total/10), (n_total/10)*2, (n_total/10)*3, (n_total/10)*4, (n_total/10)*5, (n_total/10)*6, 
         (n_total/10)*7, (n_total/10)*8, (n_total/10)*9, sys.maxsize],
   labels=['10', '20', '30', '40', '50', '60', '70', '80', '90', '100']
)

# Certificar-se de que a variável y está no formato numérico
df_age_cm["y"] = pd.to_numeric(df_age_cm["y"], errors='coerce')

# Adicionando cálculo de n, min_age, max_age, média de survival e logito_p
idade_sum_sm = df_age_cm.groupby('percentil_idade').agg(
    n=('age', 'count'),                     # Número de linhas
    min_age=('age', 'min'),                 # Mínimo de Age
    max_age=('age', 'max'),                 # Máximo de Age
    mean_y=('y', 'mean')      # Média de Survival
).reset_index()

# %%
idade_sum_sm
# Calculando logito_p
epsilon = 1e-10  # Para evitar 0 e 1
idade_sum_sm['logito_p'] = np.log(np.clip(idade_sum_sm['mean_y'], epsilon, 1 - epsilon) / 
                                  (1 - np.clip(idade_sum_sm['mean_y'], epsilon, 1 - epsilon)))

# Exibindo a tabela resultante
idade_sum_sm

# fazendo o gráfico de dispersão para p
sns.scatterplot(x="percentil_idade",y="mean_y",data=idade_sum_sm)
idade_sum_sm

# %%



#############################################
#  CRIANDO OS GRUPOS DE IDADE
#############################################


# 1. Converte y para numérico (caso ainda não esteja)
bank['y'] = bank['y'].map({'yes': 1, 'no': 0})

# 3. Recrie a variável Age_C garantindo que cobre todo o intervalo (exemplo dinâmico baseado na tabela anterior):
bank['Age_C'] = pd.cut(
    bank['age'],
    bins=[-float('inf'), 29, 56, float('inf')],
    labels=['Ate_29', '29_a_56', 'Acima_56'],
    ordered=False
)

# %%
# 4. Rode o agrupamento mostrando a contagem de 0s, 1s, total e a taxa
resultado = bank.groupby('Age_C', observed=False).agg(
    qtd_0 = ('y', lambda x: (x == 0).sum()),
    qtd_1 = ('y', lambda x: (x == 1).sum()),
    n = ('y', 'count'),
    tx_resposta = ('y', 'mean')
)

print(resultado)

# %%


#############################################
#    BALANCE
#############################################

# criando uma copia para manipular o que importa
df_duration_cm = bank[['y', 'balance']].copy()
# transformando em 1 e 0 
df_duration_cm['y'] = df_duration_cm['y'].map({'yes': 1, 'no': 0})

# %%
# 1. Ordenando pelo balance (conforme o seu objetivo)
df_duration_cm = df_duration_cm.sort_values("balance").reset_index(drop=True)
# 2. Criando uma variável com o n_total da base
n_total = len(df_duration_cm)
# 3. Criando os limites (bins) de forma limpa usando o n_total
bins = [-1] + [(n_total / 10) * i for i in range(1, 10)] + [n_total]
# 4. Criando a variável percentil do balance, de 10 em 10%
df_duration_cm['percentil_balance'] = pd.cut(
    df_duration_cm.index,
    bins=bins,
    labels=['10', '20', '30', '40', '50', '60', '70', '80', '90', '100'],
    include_lowest=True
)

# Certificar-se de que a variável y está no formato numérico
df_duration_cm["y"] = pd.to_numeric(df_duration_cm["y"], errors='coerce')
# Adicionando cálculo de n, min_age, max_age, média de survival e logito_p
balance_sum_sm = df_duration_cm.groupby('percentil_balance').agg(
    n=('balance', 'count'),                     # Número de linhas
    min_age=('balance', 'min'),                 # Mínimo de Age
    max_age=('balance', 'max'),                 # Máximo de Age
    mean_y=('y', 'mean')      # Média de Survival
).reset_index()
balance_sum_sm
# Calculando logito_p
epsilon = 1e-10  # Para evitar 0 e 1
balance_sum_sm['logito_p'] = np.log(np.clip(balance_sum_sm['mean_y'], epsilon, 1 - epsilon) / 
                                  (1 - np.clip(balance_sum_sm['mean_y'], epsilon, 1 - epsilon)))
# Exibindo a tabela resultante
balance_sum_sm
# fazendo o gráfico de dispersão para p
sns.scatterplot(x="percentil_balance",y="mean_y",data=balance_sum_sm)
balance_sum_sm


# %%


#############################################
#    DURATION
#############################################

# criando uma copia para manipular o que importa
df_duration_cm = bank[['y', 'duration']].copy()
# transformando em 1 e 0 
df_duration_cm['y'] = df_duration_cm['y'].map({'yes': 1, 'no': 0})

# %%
# 1. Ordenando pelo balance (conforme o seu objetivo)
df_duration_cm = df_duration_cm.sort_values("duration").reset_index(drop=True)
# 2. Criando uma variável com o n_total da base
n_total = len(df_duration_cm)
# 3. Criando os limites (bins) de forma limpa usando o n_total
bins = [-1] + [(n_total / 10) * i for i in range(1, 10)] + [n_total]
# 4. Criando a variável percentil do balance, de 10 em 10%
df_duration_cm['percentil_duration'] = pd.cut(
    df_duration_cm.index,
    bins=bins,
    labels=['10', '20', '30', '40', '50', '60', '70', '80', '90', '100'],
    include_lowest=True
)

# Certificar-se de que a variável y está no formato numérico
df_duration_cm["y"] = pd.to_numeric(df_duration_cm["y"], errors='coerce')
# Adicionando cálculo de n, min_age, max_age, média de survival e logito_p
balance_sum_sm = df_duration_cm.groupby('percentil_duration').agg(
    n=('duration', 'count'),                     # Número de linhas
    min_age=('duration', 'min'),                 # Mínimo de Age
    max_age=('duration', 'max'),                 # Máximo de Age
    mean_y=('y', 'mean')      # Média de Survival
).reset_index()
balance_sum_sm
# Calculando logito_p
epsilon = 1e-10  # Para evitar 0 e 1
balance_sum_sm['logito_p'] = np.log(np.clip(balance_sum_sm['mean_y'], epsilon, 1 - epsilon) / 
                                  (1 - np.clip(balance_sum_sm['mean_y'], epsilon, 1 - epsilon)))
# Exibindo a tabela resultante
balance_sum_sm
# fazendo o gráfico de dispersão para p
sns.scatterplot(x="percentil_duration",y="mean_y",data=balance_sum_sm)
balance_sum_sm

# %%
bank

# %%
