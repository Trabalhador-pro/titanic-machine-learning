# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

# Carrega a base de da criação de modelo
df_train = pd.read_csv("data/train.csv")

# Carregando a predisão do desafio 
df_test_final = pd.read_csv('data/test.csv')

# Olhando e entendendo os dados
df_train.head()

# %%
# Analisando os tipos de dados e valores faltantes para planejamneto de limpeza 

# Olhando os tipos de dados e se tem valores ausentes
df_train.info()

# Aqui olhamos a quantidade de valores ausentes em cada coluna 
df_train.isna().sum()

# %%
# Separando colunas que não tem valores preditivos e que não serão ser utilizadas

# Tirando algumas colunas que nao vou utilizar 
df2 = df_train.drop(columns=['Cabin', 'Ticket', 'Name'])

# Define as variáveis preditoras
features = df2.drop(columns=["Survived", "PassengerId"])

# Define a variável alvo
target = df2["Survived"]

# %%
# Criando o um Pipeline De limpeza de dados

# Seprando colunas categoricas e numericas 
colunas_numericas = features.select_dtypes(include=['float64','int64']).columns.to_list()
colunas_categoricas = features.select_dtypes(include=['object']).columns.to_list()

# Criando um tratamento para valores ausentes de dados numericos
Pipeline_numerico = Pipeline([
    ('inputer', SimpleImputer(strategy='mean'))
])

# Traamento de valores ausentes para dados categoricos e fazendo a conversao para numericos
Pipeline_categorico = Pipeline([
    ('inputer', SimpleImputer(strategy='most_frequent')),

    ('encoder', OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ))
])

# Criando o pepiline para o tratamento final 
preprocessador = ColumnTransformer([
    ('num', Pipeline_numerico, colunas_numericas),
    ('cat', Pipeline_categorico, colunas_categoricas) 
])

# Criando o Pipeline final com o modelo
modelo_pipeline = Pipeline([
    ('preprocessamento', preprocessador),
    ('modelo', tree.DecisionTreeClassifier(max_depth=3, random_state=42))
])

# %%
# Separando treino e teste
X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=target
)

# Verificando a taxa de resposta
print("Taxa geral:", df2['Survived'].mean())
print("Taxa treino:", y_train.mean())
print("Taxa teste:", y_test.mean())

# %%
modelo_pipeline.fit(X_train, y_train)

# Fazendo as previsões dos treinos e teste
y_pred_train = modelo_pipeline.predict(X_train)
y_pred_test = modelo_pipeline.predict(X_test)

# Avalia desempenho no treino
acc_train = metrics.accuracy_score(y_train, y_pred_train)
precision_train = metrics.precision_score(y_train, y_pred_train)

# Avalia desempenho no teste
acc_test = metrics.accuracy_score(y_test, y_pred_test)
precision_test = metrics.precision_score(y_test, y_pred_test)

# Mostra o desepenho do treino
print('Acurácia treino', acc_train)
print('Precisão', precision_train)
# Mostra o desenpenho do teste
print('Acurácia teste', acc_test)
print('Precisão', precision_test)

# %%
# Importando o arquirvo de teste do desafio
dt_final_teste = pd.read_csv('data/test.csv')

# Para criar a tabela final de predicao
passenger_id = dt_final_teste['PassengerId']

# Limpando as colunas que foram limpas antes no começo
X_final = df_test_final.drop(columns=["Cabin", "Ticket", "Name", "PassengerId"])

# Fazendo a previsao dos dados do teste
previsao_final = modelo_pipeline.predict(X_final)

# %%
# Criando a tabela pedida pelo kaggle para a avalição
resultado = pd.DataFrame({
    'PassengerId': passenger_id,
    'Survived': previsao_final
})

# Criando um arquivo
resultado.to_csv("submission.csv", index=False)