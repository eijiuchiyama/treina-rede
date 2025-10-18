import pandas as pd

# Lê o CSV original
df = pd.read_csv('tabela_tratada.csv')

# Pega as últimas N linhas (ex: 10)
ultimas_linhas = df.tail(10)

# Salva essas linhas em outro CSV
ultimas_linhas.to_csv('dados_validar.csv', index=False)
