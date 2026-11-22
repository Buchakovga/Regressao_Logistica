
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
from sklearn.metrics import confusion_matrix, precision_score, recall_score,roc_auc_score, roc_curve



# %%

#assim como na regressão linear, criamos funções personalizadas para os métodos de seleção de variáveis
#carregando funções, todas de uma vez
from funcoes_importantes_regressao_logistica import step,univariada_variavel_numerica,univariada_variavel_categoria,analise_var_numerica_por_percentil


# %%

#importando base bank
bank = pd.read_csv("dados/bank.csv",sep=";")
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
univariada_variavel_categoria(bank, "y")

# %%


#######################################################################################
##############     ANALISE BIVARIADA DAS VARIÁVEIS NUMERICAS             ##############
#######################################################################################


#############################################
#  BIVARIADA PARA   IDADE
#############################################


# criando uma copia para manipular o que importa
df_age_cm = bank[['y', 'age']].copy()
# transformando em 1 e 0 
# Mapeia explicitamente 'yes' para 1 e 'no' para 0
df_age_cm['y'] = df_age_cm['y'].map({'yes': 1, 'no': 0})
df_age_cm

# ordenando pela Idade
df_age_cm = df_age_cm.sort_values("age").reset_index()
# criando uma variavel com o n_total da base
n_total = df_age_cm['y'].count()

df_age_cm["y"] = pd.to_numeric(df_age_cm["y"], errors="coerce")
df_age_cm["percentil_idade"] = pd.qcut(
    df_age_cm["age"],  # substitua pelo nome real da coluna de idade
    q=10,
    labels=["10", "20", "30", "40", "50", "60", "70", "80", "90", "100"],
    duplicates="drop",  # evita erro se houver muitos valores empatados
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



#############################################
#  CRIANDO OS GRUPOS DE IDADE
#############################################

# %%

# 1. Converte y para numérico (caso ainda não esteja)
bank['y'] = bank['y'].map({'yes': 1, 'no': 0})
# %%
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
#   BIVARIADA PARA  BALANCE
#############################################


# Criando uma cópia para manipular o que importa
df_balance_cm = bank[['y', 'balance']].copy()

# Transformando em 1 e 0 de forma segura (funciona se já for 0/1 ou 'yes'/'no')
if (
    df_balance_cm['y'].dtype == 'object'
    or str(df_balance_cm['y'].dtype) == 'category'
):
  df_balance_cm['y'] = (
      df_balance_cm['y'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
  )
else:
  df_balance_cm['y'] = pd.to_numeric(df_balance_cm['y'], errors='coerce')

# 1. Ordenando pelo balance
df_balance_cm = df_balance_cm.sort_values('balance').reset_index(drop=True)

# 2. Criando uma variável com o n_total da base
n_total = len(df_balance_cm)

# 3. Criando os limites (bins) de forma limpa usando o n_total
bins = [-1] + [(n_total / 10) * i for i in range(1, 10)] + [n_total]

# 4. Criando a variável percentil do balance, de 10 em 10%
df_balance_cm['percentil_balance'] = pd.cut(
    df_balance_cm.index,
    bins=bins,
    labels=['10', '20', '30', '40', '50', '60', '70', '80', '90', '100'],
    include_lowest=True,
)

# Agrupando e calculando n, min_balance, max_balance e média de y
balance_sum_sm = (
    df_balance_cm.groupby('percentil_balance', observed=False)
    .agg(
        n=('balance', 'count'),
        min_balance=('balance', 'min'),
        max_balance=('balance', 'max'),
        mean_y=('y', 'mean'),
    )
    .reset_index()
)

# Calculando logito_p
epsilon = 1e-10  # Para evitar 0 e 1
p_clip = np.clip(balance_sum_sm['mean_y'], epsilon, 1 - epsilon)
balance_sum_sm['logito_p'] = np.log(p_clip / (1 - p_clip))

# Exibindo a tabela resultante
display(balance_sum_sm)

# Fazendo o gráfico de dispersão para p
sns.scatterplot(x='percentil_balance', y='mean_y', data=balance_sum_sm)



# %%


#############################################
#   BIVARIADA PARA  DURATION
#############################################

# Criando uma cópia para manipular o que importa
df_duration_cm = bank[['y', 'duration']].copy()

# Transformando em 1 e 0 de forma segura (funciona se já for 0/1 ou 'yes'/'no')
if (
    df_duration_cm['y'].dtype == 'object'
    or str(df_duration_cm['y'].dtype) == 'category'
):
  df_duration_cm['y'] = (
      df_duration_cm['y']
      .astype(str)
      .str.strip()
      .str.lower()
      .map({'yes': 1, 'no': 0})
  )
else:
  df_duration_cm['y'] = pd.to_numeric(df_duration_cm['y'], errors='coerce')

# 1. Ordenando pelo duration
df_duration_cm = df_duration_cm.sort_values('duration').reset_index(drop=True)

# 2. Criando uma variável com o n_total da base
n_total = len(df_duration_cm)

# 3. Criando os limites (bins) de forma limpa usando o n_total
bins = [-1] + [(n_total / 10) * i for i in range(1, 10)] + [n_total]

# 4. Criando a variável percentil do duration, de 10 em 10%
df_duration_cm['percentil_duration'] = pd.cut(
    df_duration_cm.index,
    bins=bins,
    labels=['10', '20', '30', '40', '50', '60', '70', '80', '90', '100'],
    include_lowest=True,
)

# Agrupando e calculando n, min_duration, max_duration e média de y
duration_sum_sm = (
    df_duration_cm.groupby('percentil_duration', observed=False)
    .agg(
        n=('duration', 'count'),
        min_duration=('duration', 'min'),
        max_duration=('duration', 'max'),
        mean_y=('y', 'mean'),
    )
    .reset_index()
)

# Calculando logito_p
epsilon = 1e-10
p_clip = np.clip(duration_sum_sm['mean_y'], epsilon, 1 - epsilon)
duration_sum_sm['logito_p'] = np.log(p_clip / (1 - p_clip))

# Exibindo a tabela resultante
display(duration_sum_sm)

# Fazendo o gráfico de dispersão para p
sns.scatterplot(x='percentil_duration', y='mean_y', data=duration_sum_sm)

# %%



#############################################
#    BIVARIADA PARA CAMPAIGN
#############################################


# Criando uma cópia para manipular o que importa
df_campaign_cm = bank[['y', 'campaign']].copy()

# Converte 'yes'/'no' (mesmo em StringArray do Pandas 3) para float/int 1 e 0
df_campaign_cm['y'] = pd.to_numeric(
    df_campaign_cm['y'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0}),
    errors='coerce'
)

# 1. Ordenando pelo campaign
df_campaign_cm = df_campaign_cm.sort_values('campaign').reset_index(drop=True)

# 2. Criando uma variável com o n_total da base
n_total = len(df_campaign_cm)

# 3. Criando os limites (bins)
bins = [-1] + [(n_total / 10) * i for i in range(1, 10)] + [n_total]

# 4. Criando a variável percentil do campaign, de 10 em 10%
df_campaign_cm['percentil_campaign'] = pd.cut(
    df_campaign_cm.index,
    bins=bins,
    labels=['10', '20', '30', '40', '50', '60', '70', '80', '90', '100'],
    include_lowest=True,
)

# Agrupando e calculando n, min_campaign, max_campaign e média de y
campaign_sum_sm = (
    df_campaign_cm.groupby('percentil_campaign', observed=False)
    .agg(
        n=('campaign', 'count'),
        min_campaign=('campaign', 'min'),
        max_campaign=('campaign', 'max'),
        mean_y=('y', 'mean'),
    )
    .reset_index()
)

# Calculando logito_p
epsilon = 1e-10
p_clip = np.clip(campaign_sum_sm['mean_y'], epsilon, 1 - epsilon)
campaign_sum_sm['logito_p'] = np.log(p_clip / (1 - p_clip))

# Exibindo a tabela resultante
display(campaign_sum_sm)

# Fazendo o gráfico de dispersão para p
sns.scatterplot(x='percentil_campaign', y='mean_y', data=campaign_sum_sm)


# %%

#############################################
#    BIVARIADA PARA PDAYS
#############################################


# Criando uma cópia para manipular o que importa
df_pdays_cm = bank[['y', 'pdays']].copy()

# Converte 'yes'/'no' (compatível com StringArray do Pandas 3) para numérico 1 e 0
df_pdays_cm['y'] = pd.to_numeric(
    df_pdays_cm['y'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0}),
    errors='coerce'
)

# 1. Ordenando pelo pdays
df_pdays_cm = df_pdays_cm.sort_values('pdays').reset_index(drop=True)

# 2. Criando uma variável com o n_total da base
n_total = len(df_pdays_cm)

# 3. Criando os limites (bins) de forma limpa usando o n_total
bins = [-1] + [(n_total / 10) * i for i in range(1, 10)] + [n_total]

# 4. Criando a variável percentil do pdays, de 10 em 10%
df_pdays_cm['percentil_pdays'] = pd.cut(
    df_pdays_cm.index,
    bins=bins,
    labels=['10', '20', '30', '40', '50', '60', '70', '80', '90', '100'],
    include_lowest=True,
)

# Agrupando e calculando n, min_pdays, max_pdays e média de y
pdays_sum_sm = (
    df_pdays_cm.groupby('percentil_pdays', observed=False)
    .agg(
        n=('pdays', 'count'),
        min_pdays=('pdays', 'min'),
        max_pdays=('pdays', 'max'),
        mean_y=('y', 'mean'),
    )
    .reset_index()
)

# Calculando logito_p
epsilon = 1e-10
p_clip = np.clip(pdays_sum_sm['mean_y'], epsilon, 1 - epsilon)
pdays_sum_sm['logito_p'] = np.log(p_clip / (1 - p_clip))

# Exibindo a tabela resultante
display(pdays_sum_sm)

# Fazendo o gráfico de dispersão para p
sns.scatterplot(x='percentil_pdays', y='mean_y', data=pdays_sum_sm)


# %%

#############################################
#   CRIANDO VARIÁVEL CATEGÓRICA PARA PDAYS
#############################################



# 1. Converte y para numérico de forma segura (funciona se já for 0/1 ou 'yes'/'no')
if not pd.api.types.is_numeric_dtype(bank['y']):
  bank['y'] = pd.to_numeric(
      bank['y'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0}),
      errors='coerce',
  )

# 2. Cria a variável categórica Pdays_C com as 3 faixas:
# (-inf, 0]   -> pega o -1 (e eventual 0) = '0_dias' (nunca contatado)
# (0, 200]    -> pega de 1 até 200 dias    = 'Ate_200_dias'
# (200, inf)  -> pega acima de 200 dias    = 'Acima_200_dias'
bank['Pdays_C'] = pd.cut(
    bank['pdays'],
    bins=[-float('inf'), 0, 200, float('inf')],
    labels=['0_dias', 'Ate_200_dias', 'Acima_200_dias'],
    ordered=False,
)

# 3. Roda o agrupamento mostrando a contagem de 0s, 1s, total e a taxa
resultado_pdays = bank.groupby('Pdays_C', observed=False).agg(
    qtd_0=('y', lambda x: (x == 0).sum()),
    qtd_1=('y', lambda x: (x == 1).sum()),
    n=('y', 'count'),
    tx_resposta=('y', 'mean'),
)
print(resultado_pdays)

# %%


#############################################
#    BIVARIADA PARA PREVIOUS
#############################################


df_previous_cm = bank[['y', 'previous']].copy()
df_previous_cm['y'] = pd.to_numeric(df_previous_cm['y'], errors='coerce')

# 1. Ordenar por previous e repor o índice
df_previous_cm = df_previous_cm.sort_values('previous').reset_index(drop=True)

# 2. Total de linhas
n_total = len(df_previous_cm)

# 3. Limites dos 10 decis
bins = [-1] + [(n_total / 10) * i for i in range(1, 10)] + [n_total]

# 4. Criar a variável de percentis (10 a 100)
df_previous_cm['percentil_previous'] = pd.cut(
    df_previous_cm.index,
    bins=bins,
    labels=['10', '20', '30', '40', '50', '60', '70', '80', '90', '100'],
    include_lowest=True,
)

# Agrupar estatísticas por percentil
previous_sum_sm = (
    df_previous_cm.groupby('percentil_previous', observed=False)
    .agg(
        n=('previous', 'count'),
        min_previous=('previous', 'min'),
        max_previous=('previous', 'max'),
        mean_y=('y', 'mean'),
    )
    .reset_index()
)

# Calcular o logito de p
epsilon = 1e-10
p_clip = np.clip(previous_sum_sm['mean_y'], epsilon, 1 - epsilon)
previous_sum_sm['logito_p'] = np.log(p_clip / (1 - p_clip))

display(previous_sum_sm)

# Gráfico de dispersão
sns.scatterplot(x='percentil_previous', y='mean_y', data=previous_sum_sm)

# %%

#############################################
#   CRIANDO VARIÁVEL CATEGÓRICA PARA PREVIOUS
#############################################


# 1. Converte y para numérico de forma segura (funciona se já for 0/1 ou 'yes'/'no')
if not pd.api.types.is_numeric_dtype(bank['y']):
  bank['y'] = pd.to_numeric(
      bank['y'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0}),
      errors='coerce',
  )

# 2. Cria a variável categórica Previous_C com 2 faixas:
# (-inf, 0]  -> previous == 0 -> '0'
# (0, inf)   -> previous > 0  -> 'Mais_que_0'
bank['Previous_C'] = pd.cut(
    bank['previous'],
    bins=[-float('inf'), 0, float('inf')],
    labels=['0', 'Mais_que_0'],
    ordered=False,
)

# 3. Roda o agrupamento mostrando a contagem de 0s, 1s, total e a taxa
resultado_previous = bank.groupby('Previous_C', observed=False).agg(
    qtd_0=('y', lambda x: (x == 0).sum()),
    qtd_1=('y', lambda x: (x == 1).sum()),
    n=('y', 'count'),
    tx_resposta=('y', 'mean'),
)
print(resultado_previous)

# %%



############################################################################
########### ANALISANDO A DISTRIBUIÇÃO DAS CATEGÓRICAS COM Y   ##############
############################################################################


#Para a variavel Age_C
sns.countplot(data=bank, x='Age_C', hue='y', dodge=True)

# %%

#Para a variavel Pdays_C
sns.countplot(data=bank, x='Pdays_C', hue='y', dodge=True)

# %%

#Para a variavel Previous_C
sns.countplot(data=bank, x='Previous_C', hue='y', dodge=True)

# %%

#Para a variavel job

# Ordem das categorias do maior para o menor volume
ordem_job = bank['job'].value_counts().index

plt.figure(figsize=(10, 6))
sns.countplot(data=bank, y='job', hue='y', order=ordem_job, dodge=True)
plt.title('Distribuição de y por job')
plt.xlabel('Quantidade')
plt.ylabel('job')
plt.tight_layout()
plt.show()


#######
#### CRIANDA A CATEGORICA PARA O JOB. 
######
 
# %%

# 1. Converte y para numérico de forma segura (funciona se já for 0/1 ou 'yes'/'no')
if not pd.api.types.is_numeric_dtype(bank['y']):
  bank['y'] = pd.to_numeric(
      bank['y'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0}),
      errors='coerce',
  )

# 2. Cria a variável categórica Job_C agrupando as profissões
job_limpo = bank['job'].astype(str).str.strip().str.lower()

mapa_job = {
    'management': 'Management_Technician',
    'technician': 'Management_Technician',
    'blue-collar': 'BlueCollar_Admin_Retired',
    'admin.': 'BlueCollar_Admin_Retired',
    'retired': 'BlueCollar_Admin_Retired',
}

bank['Job_C'] = job_limpo.map(mapa_job).fillna('Outros')

# 3. Roda o agrupamento mostrando a contagem de 0s, 1s, total e a taxa
resultado_job = bank.groupby('Job_C', observed=False).agg(
    qtd_0=('y', lambda x: (x == 0).sum()),
    qtd_1=('y', lambda x: (x == 1).sum()),
    n=('y', 'count'),
    tx_resposta=('y', 'mean'),
)
print(resultado_job)


# %%



#Para a variavel marital
sns.countplot(data=bank, x='marital', hue='y', dodge=True)


# %%

#Para a variavel education
sns.countplot(data=bank, x='education', hue='y', dodge=True)


# %%

#Para a variavel housing
sns.countplot(data=bank, x='housing', hue='y', dodge=True)


# %%

#Para a variavel loan
sns.countplot(data=bank, x='loan', hue='y', dodge=True)


# %%

#Para a variavel contact
sns.countplot(data=bank, x='contact', hue='y', dodge=True)



# %%
##########################################################################
####### CRIANDO AS VARIÁVEIS DUMMIES                        ##############
##########################################################################

colunas_num = ['campaign']  
colunas_cat = [
    'Job_C',
    'marital',
    'education',
    'default',
    'housing',
    'loan',
    'contact',
    'Pdays_C',
    'Previous_C',
]

bank_modelo = bank[['y'] + colunas_num + colunas_cat].copy()

# Usa o nome da variável no prefixo para evitar nomes repetidos
prefixos = {c: f'dummie_{c.lower()}' for c in colunas_cat}

bank_modelo = pd.get_dummies(
    bank_modelo, prefix=prefixos, columns=colunas_cat, dtype=np.int64
)

bank_modelo.columns = (
    bank_modelo.columns.str.lower()
    .str.replace(' ', '_', regex=False)
    .str.replace('-', '_', regex=False)
    .str.replace('.', '', regex=False)
)

print(bank_modelo.columns.tolist())
bank_modelo

# %%

###########################################################
########           RODANDO O MODELO             ###########
###########################################################


# 1. Variável resposta y
y = pd.to_numeric(bank_modelo['y'], errors='coerce')

# 2. Remove 'y' e uma categoria de referência por variável categórica
colunas_drop = [
    'y',
    'dummie_job_c_outros',
    'dummie_job_c_bluecollar_admin_retired',
    'dummie_job_c_management_technician',    
    'dummie_marital_single',
    'dummie_education_unknown',
    'dummie_education_primary',
    'dummie_education_secondary',
    'dummie_education_tertiary',
    'dummie_default_no',
    'dummie_default_yes',
    'dummie_housing_no',
    'dummie_loan_no',
    'dummie_contact_unknown',
    'dummie_pdays_c_0_dias',
    'dummie_previous_c_0',
]

X = bank_modelo.drop(columns=colunas_drop).copy()
X = X.apply(pd.to_numeric, errors='coerce')

# Garante alinhamento e ausência de NaN/inf
dados_limpos = pd.concat([y.rename('y'), X], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
y_clean = dados_limpos['y']
X_clean = sm.add_constant(dados_limpos.drop(columns=['y']), has_constant='add')

# 3. Ajuste da regressão logística (GLM Binomial)
model_bank = sm.GLM(y_clean, X_clean, family=sm.families.Binomial()).fit()
print(model_bank.summary())



# %%

# 4. Tabela consolidada com Coeficiente, Odds Ratio (OR), IC 95% e p-valor
ic = model_bank.conf_int()
tabela_or = pd.DataFrame({
    'OR': np.exp(model_bank.params),
    'IC_2.5%': np.exp(ic[0]),
    'IC_97.5%': np.exp(ic[1]),
    'z': model_bank.tvalues,
    'P>|z|': model_bank.pvalues,
})

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)

print(
    tabela_or.to_string(
        formatters={
            'OR': lambda x: f'{x:.4f}',
            'IC_2.5%': lambda x: f'{x:.4f}',
            'IC_97.5%': lambda x: f'{x:.4f}',
            'z': lambda x: f'{x:.2f}',
            'P>|z|': lambda x: '<0.0001' if x < 0.0001 else f'{x:.4f}',
        }
    )
)

# %%


#################################################################################
######               CRIANDO AS MÉTRICAS                ######################### 
#################################################################################

# CRIANDO O Y-CHAPEU
bank_metricas = bank_modelo.copy()
bank_metricas['y_chapeu'] =model_bank.predict(X_clean)
bank_metricas = bank_metricas[['y','y_chapeu']]

# %%


# CRIANDO A REGRA DE DECISÃO
# 1. Regras de decisão sobre a probabilidade prevista (usa 'prob_y' se existir, senão 'y_chapeu')
base_prob = (
    bank_metricas['prob_y']
    if 'prob_y' in bank_metricas.columns
    else bank_metricas['y_chapeu']
)

cortes = {
    'decisao_003': 0.03,  # 3%
    'decisao_005': 0.05,  # 5%
    'decisao_010': 0.10,  # 10%
    'decisao_020': 0.20,  # 20%
    'decisao_025': 0.25,  # 25%
}

for col, limite in cortes.items():
  bank_metricas[col] = np.where(base_prob >= limite, 1.0, 0.0)

# 2. Resumo de contagem e percentual para cada regra
for col, limite in cortes.items():
  contagem = bank_metricas[col].value_counts(dropna=False).reindex([0.0, 1.0], fill_value=0)
  percentual = (contagem / len(bank_metricas)) * 100

  resumo = pd.DataFrame({
      'quantidade': contagem,
      'percentual_%': percentual.round(2),
  })

  print(f'\n=== Regra {col} (corte >= {limite:.0%}) ===')
  display(resumo)


# %%

import importlib
import decisao_func
importlib.reload(decisao_func)
from decisao_func import avaliar_decisao


# %%

matriz_010, resumo_010 = avaliar_decisao(bank_metricas, decisao='decisao_010')

# %%


# Garante que decisao_020 existe (caso tenha sido criada sobre prob_y ou y_chapeu)
if 'decisao_010' not in bank_metricas.columns:
  base_prob = (
      bank_metricas['prob_y']
      if 'prob_y' in bank_metricas.columns
      else bank_metricas['y_chapeu']
  )
  bank_metricas['decisao_020'] = np.where(base_prob >= 0.20, 1, 0)

y_true = bank_metricas['y'].astype(int)
y_pred = bank_metricas['decisao_020'].astype(int)

# Matriz de confusão (linhas = real, colunas = previsto: 0 e 1)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
matriz_df = pd.DataFrame(
    cm,
    index=['Real = 0', 'Real = 1'],
    columns=['Previsto = 0', 'Previsto = 1'],
)

print('=== Matriz de Confusão (Corte >= 0.10) ===')
display(matriz_df)

# Extrai VN, FP, FN, VP e calcula as métricas principais
vn, fp, fn, vp = cm.ravel()
precisao = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)

resumo_metricas = pd.DataFrame({
    'VN (0 real / 0 prev)': [vn],
    'FP (0 real / 1 prev)': [fp],
    'FN (1 real / 0 prev)': [fn],
    'VP (1 real / 1 prev)': [vp],
    'Precisão': [round(precisao, 4)],
    'Sensibilidade / Recall': [round(recall, 4)],
})

display(resumo_metricas)

# %%

bank_metricas

# %%


col_real = 'y_real' if 'y_real' in bank_metricas.columns else 'y'
y_true = pd.to_numeric(bank_metricas[col_real], errors='coerce').astype(int)
y_score = pd.to_numeric(bank_metricas['y_chapeu'], errors='coerce')

df_roc = pd.DataFrame({'y_true': y_true, 'y_chapeu': y_score}).dropna()

fpr, tpr, thresholds = roc_curve(df_roc['y_true'], df_roc['y_chapeu'])
auc_roc = roc_auc_score(df_roc['y_true'], df_roc['y_chapeu'])

print(f'AUC-ROC: {auc_roc:.4f}')

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f'Curva ROC (AUC = {auc_roc:.4f})', linewidth=2)
plt.plot(
    [0, 1], [0, 1], linestyle='--', color='gray', label='Aleatório (AUC = 0.50)'
)
plt.xlabel('1 - Especificidade (FPR)')
plt.ylabel('Sensibilidade / Recall (TPR)')
plt.title('Curva ROC — Regressão Logística')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %%


################################################
######           Calculando o MSE    ###########
################################################
bank_metricas['MSE'] = (bank_metricas['y'] - bank_metricas['y_chapeu'])**2
MSE_modelo = bank_metricas['MSE'].mean()
print("MSE do modelo:", MSE_modelo)
#Calculando o RMSE
RMSE_modelo = np.sqrt(MSE_modelo)
print("RMSE do modelo:", RMSE_modelo)

# %%

bank_metricas

# %%


import numpy as np
import pandas as pd

# 1. Garante coluna real ('y_real' ou 'y') e probabilidade prevista ('y_chapeu' ou 'prob_y')
col_real = 'y_real' if 'y_real' in bank_metricas.columns else 'y'
col_prob = 'prob_y' if 'prob_y' in bank_metricas.columns else 'y_chapeu'

df_cal = pd.DataFrame({
    'y_true': pd.to_numeric(bank_metricas[col_real], errors='coerce'),
    'prob_prevista': pd.to_numeric(bank_metricas[col_prob], errors='coerce'),
}).dropna()

# 2. Define faixas de probabilidade (com foco fino em torno de 15%)
bins = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.01]
labels = [
    '0% – 5%',
    '5% – 10%',
    '10% – 15%',
    '15% – 20%',
    '20% – 30%',
    '30% – 50%',
    '50% – 100%',
]

df_cal['faixa_prob'] = pd.cut(
    df_cal['prob_prevista'], bins=bins, labels=labels, right=False
)

# 3. Agrega quantidade de casos, média prevista e taxa real observada
tabela_calibracao = (
    df_cal.groupby('faixa_prob', observed=False)
    .agg(
        qtd_clientes=('y_true', 'count'),
        eventos_reais=('y_true', 'sum'),
        prob_media_prevista=('prob_prevista', 'mean'),
        taxa_real_observada=('y_true', 'mean'),
    )
    .reset_index()
)

tabela_calibracao['prob_media_%'] = tabela_calibracao['prob_media_prevista'] * 100
tabela_calibracao['taxa_real_%'] = tabela_calibracao['taxa_real_observada'] * 100
tabela_calibracao['diferenca_pp'] = (
    tabela_calibracao['taxa_real_%'] - tabela_calibracao['prob_media_%']
)

cols_exibir = [
    'faixa_prob',
    'qtd_clientes',
    'eventos_reais',
    'prob_media_%',
    'taxa_real_%',
    'diferenca_pp',
]

display(
    tabela_calibracao[cols_exibir].round({
        'prob_media_%': 2,
        'taxa_real_%': 2,
        'diferenca_pp': 2,
    })
)

# %%
