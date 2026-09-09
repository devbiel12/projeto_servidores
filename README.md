# Projeto Servidores

Análise estatística e classificação de servidores de TI a partir de dados armazenados em CSV.

## Funcionalidades

O programa realiza:

- Cálculo da média e da variância amostral de `uso_cpu` e `latencia_rede_ms`.
- Cálculo de um intervalo de confiança de 95% para a média de `uso_cpu` usando a distribuição t de Student.
- Construção de uma tabela de contingência entre `tipo_rede` e `status_alerta`.
- Aplicação do teste qui-quadrado de independência.
- Padronização das métricas numéricas.
- Redução de dimensionalidade para duas componentes principais usando PCA.
- Classificação de alertas com KNN usando cinco vizinhos.
- Exibição da matriz de confusão e do relatório de classificação.

## Requisitos

- Python 3.10 ou superior
- Um arquivo CSV com os dados dos servidores
- Dependências listadas em [requirements.txt](requirements.txt)

## Instalação

Clone o repositório e entre na pasta do projeto:

```bash
git clone URL_DO_REPOSITORIO
cd projeto_servidores
```

Recomenda-se criar um ambiente virtual:

```bash
python -m venv .venv
```

No Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

No Linux ou macOS:

```bash
source .venv/bin/activate
```

Instale as dependências:

```bash
python -m pip install -r requirements.txt
```

## Dados de entrada

Por padrão, o programa procura o arquivo `servidores_ti.csv` na mesma pasta do script.

O CSV deve conter as seguintes colunas:

```text
uso_cpu
uso_memoria
latencia_rede_ms
taxa_pacotes_perdidos
tipo_rede
status_alerta
```

## Execução

Com o CSV na mesma pasta:

```bash
python projeto_servidores.py
```

Também é possível informar o caminho do arquivo CSV como argumento:

```bash
python projeto_servidores.py caminho/para/servidores_ti.csv
```

No Windows PowerShell, por exemplo:

```powershell
python projeto_servidores.py .\dados\servidores_ti.csv
```

## Resultados

Durante a execução, são exibidos no terminal:

- Valores nulos por coluna.
- Médias e variâncias amostrais.
- Intervalo de confiança de 95%.
- Resultado do teste qui-quadrado.
- Variância explicada pelo PCA.
- Matriz de confusão do classificador KNN.
- Precisão, revocação e F1-score do modelo.

## Tecnologias

- Python
- Pandas
- NumPy
- SciPy
- Scikit-learn
