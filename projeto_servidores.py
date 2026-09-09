# IMPORTAÇÃO DAS BIBLIOTECAS NECESSÁRIAS PARA EXECUÇÃO DO CÓDIGO
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# CARREGANDO O CSV: use o caminho informado ou procure na pasta do script
path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name('servidores_ti.csv')
if not path.exists():
    raise FileNotFoundError(
        f'Arquivo CSV não encontrado: {path}\n'
        'Coloque servidores_ti.csv na pasta do programa ou informe o caminho como argumento.'
    )

df = pd.read_csv(path)

# IMPRIMINDO AS PRIMEIRAS LINHAS DO DF PARA VALIDAÇÃO
df.head()

#VERIFICANDO A PRESENÇA DE VALORES NULOS DENTRO DO CSV
print("\nValores Nulos:")
print(f'\n{df.isnull().sum()}')

"""# 3. Tarefas Práticas do Exercício Programa
Parte 1: Manipulação com Pandas, Estimadores e Intervalos de Confiança

1. Calcule a media amostral **(ar{X})** e variancia amostral(s²) para as variaveis uso_cpu e latencia_rede_ms
2. Construa formalmente um Intervalo de Confiança de 95% para a média verdadeira do uso_cpu de toda da população de servidores utlizando a distribuição t de Student(ou Z, justificado pelo tamanho amostral). Interprete o significado prático do intervalo obtido.
"""
# 1.CALCULO DA MEDIA AMOSTRAL E VARIANCIA AMOSTRAL DAS VARIAVEIS uso_cpu E latencia_rede_ms
uso_cpu_media = df['uso_cpu'].mean()
uso_cpu_variancia = df['uso_cpu'].var()

latencia_rede_ms_media = df['latencia_rede_ms'].mean()
latencia_rede_ms_variancia = df['latencia_rede_ms'].var()

print("---VALORES DE DISPERSÃO---")
print('Valores da variavel "uso_cpu"')
print(f"Media Amostral: {uso_cpu_media:.4f}")
print(f"Variancia Amostral: {uso_cpu_variancia:.4f}")

print('\nValores da variavel "latencia_rede_ms"')
print(f"Media Amostral: {latencia_rede_ms_media:.4f}")
print(f"Variancia Amostral: {latencia_rede_ms_variancia:.4f}")

# 2. CONSTRUÇÃO DO INTERVALO DE CONFIANÇA (95%)
# DADOS PARA CONSTRUÇÃO DO INTERVALO
n = len(df['uso_cpu'])
desvio_padrao = df['uso_cpu'].std()
confianca = 0.95

# Cálculo do Erro Padrão da Média
erro_padrao = desvio_padrao / np.sqrt(n)

# Cálculo do Intervalo usando a distribuição t de Student
intervalo_confianca = stats.t.interval(confianca, df=n-1, loc=uso_cpu_media, scale=erro_padrao)

print(f"--- Intervalo de Confiança para a Média (uso_cpu) ---")
print(f"Tamanho da Amostra (n): {n}")
print(f"Média Amostral (x̄): {uso_cpu_media:.4f}")
print(f"Intervalo de Confiança (95%): [{intervalo_confianca[0]:.4f} ; {intervalo_confianca[1]:.4f}]")

# Parte 2: Teste de Hipóteses (\chi^2 - Qui-Quadrado de Independência)
"""
- Investigue se há dependência estatística entre o tipo de infraestrutura onde o servidor opera (tipo_rede) e
a propensão a falhas (status_alerta):
Construa uma tabela de contingência cruzando tipo_rede e status_alerta por meio da função
apropriada do pandas.
- Aplique o teste Qui-Quadrado de Independência considerando um nível de significância de lpha = 0.05. Formule claramente a hipótese nula (H_0) e a alternativa (H_1), avalie o p-valor obtido e conclua
analiticamente.
"""

# 1. CONSTRUÇÃO DA TABELA DE CONTIGENCIA
tabela_rede_alerta = pd.crosstab(df['tipo_rede'], df['status_alerta'])
print('---Tabela Contigencia---')
print(tabela_rede_alerta)

# 2. TESTE QUI-QUADRADO DE INDEPENDENCIA
qui_2, p_valor, grau_liberdade, esperados = stats.chi2_contingency(tabela_rede_alerta)

print(f"\n---RESULTADOS DO TESTES QUI QUADRADO---")
print(f'Qui-Quadrado: ', qui_2)
print(f'P-valor: ',p_valor)
print(f'Grau de Liberdade: ', grau_liberdade)

'''
- HIPOTESE NULA(H_0): As variaveis tipo_rede e status_alerta são independentes.
- HIPOTESE ALTERNATIVA(H_1): As variaveis tipo_rede e status_alerta
não são independentes.
'''
alpha = 0.05
if p_valor < alpha:
    print("\n A hipotese nula(H_0) é rejeitada e a hipotese alternativa(H_1) é aceita, pois há estatisticas que comprovam dependência entre as variáveis.")
else:
    print("\n A hipotese nula(H_0) é aceita e a hipotese alternativa(H_1) é rejeitada, pois não há estatisticas que comprovam dependência entre as variáveis.")

"""#Parte 3: Redução de Dimensionalidade com PCA (Principal Component Analysis)
---
###Prepare as métricas quantitativas contínuas para alimentar os modelos preditivos:
- Selecione as colunas quantitativas: uso_cpu, uso_memoria, latencia_rede_ms e
taxa_pacotes_perdidos.
- Padronize os dados utilizando escalonamento de média zero e variância unitária.
- Aplique o algoritmo PCA para projetar o espaço multidimensional em exatamente 2 Componentes
Principais (PC1 e PC2). Analise a proporção da variância explicada individual e acumulada.
"""

# 1. SELEÇÃO DAS COLUNAS QUANTITATIVAS
colunas_quantitativas = ['uso_cpu', 'uso_memoria', 'latencia_rede_ms', 'taxa_pacotes_perdidos']
X = df[colunas_quantitativas]

# 2. PADRONIZANDO OS DADOS
scaler = StandardScaler()
X_padronizado = scaler.fit_transform(X)

# 3.1 APLICAÇÃO DO PCA EM 2 COMPONENTES
pca_modelo = PCA(n_components=2)
pca_resulto = pca_modelo.fit_transform(X_padronizado)
df_pca = pd.DataFrame(data=pca_resulto, columns=['PC1', 'PC2'])

# 3.2 ANALISE DA VARIANCIA INDIVIDUAL E ACUMULADA
var_individual = pca_modelo.explained_variance_ratio_
var_acumulada = np.cumsum(var_individual)
print("---Análise da Variancia Explicada---")
print(f"Variância do PC1: {var_individual[0]:.4f} ({var_individual[0]*100:.2f}%)")
print(f"Variância do PC2: {var_individual[1]:.4f} ({var_individual[1]*100:.2f}%)")
print(f"Variância Acumulada (PC1 + PC2): {var_acumulada[1]:.4f} ({var_acumulada[1]*100:.2f}%)")

print(df_pca.head())

"""#Parte 4: Classificação Supervisada com KNN (K-Nearest Neighbors)

---

### Construa um modelo preditivo baseado nas componentes principais obtidas:
- Defina as features preditoras (X) utilizando as componentes principais do PCA e o vetor alvo (y) com a
coluna status_alerta.
- Divida o conjunto de dados em Treino (70\%) e Teste (30\%) com semente aleatória fixa (ex:
random_state=42).
- Instancie e treine um classificador KNN configurado com K = 5 vizinhos.
- Avalie o modelo gerando a matriz de confusão e o relatório de classificação completo (precisão,
revocação e F1-score).
"""

# 1.1 DEFINIÇÃO DO VETOR ALVO(x)
X_knn = df_pca[['PC1', 'PC2']]

# 1.2 DEFINIÇÃO DO VETOR ALVO(y)
y_knn = df['status_alerta']

# 2. DIVISÃO DOS DADOS EM TREINO E TESTE
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X_knn, y_knn, test_size=0.30, random_state=42)

# 3. INSTANCIANDO E TREINANDO O KNN = 5
knn_modelo = knc(n_neighbors=5)
knn_modelo.fit(X_treino, Y_treino)

# 4. AVALIAÇÃO DO MODELO
Y_pred = knn_modelo.predict(X_teste)
print("--- Matriz de Confusão ---")
matriz = confusion_matrix(Y_teste, Y_pred)
print(pd.DataFrame(matriz))

print("\n--- Relatório de Classificação ---")
print(classification_report(Y_teste, Y_pred))