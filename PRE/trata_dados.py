import pandas as pd
from datetime import datetime

#Pega todos os tipos de exames realizados
tipos_exames_set = set()

for chunk in pd.read_csv('HC_EXAMES_1.csv', usecols=['DE_EXAME'], chunksize=10000, sep='|', on_bad_lines='skip', engine='python'):
    tipos_exames_set.update(chunk['DE_EXAME'].dropna().unique())

tipos_exames = sorted(tipos_exames_set)

#Categoriza os tipos de exames realizados em numérico ou binário
binarios = []
numericos = []

resultados_por_exame = {}

for chunk in pd.read_csv( #Pega o primeiro valor de cada tipo de exame para verificar se é numérico ou binário
    'HC_EXAMES_1.csv',
    sep='|',
    usecols=['DE_EXAME', 'DE_RESULTADO'],
    chunksize=10000,
    on_bad_lines='skip',
    engine='python'
):
    for _, row in chunk.iterrows():
        exame = row['DE_EXAME']
        resultado = row['DE_RESULTADO']
        
        if pd.notna(exame) and pd.notna(resultado):
            if exame not in resultados_por_exame:
                resultados_por_exame[exame] = str(resultado).strip()
                
        if len(resultados_por_exame) > len(tipos_exames):
        	break
  
def numerico(valor):
    try:
        float(str(valor).replace(',', '.'))
        return True
    except ValueError:
        return False
 
for exame, resultado in resultados_por_exame.items():
	resultado_l = str(resultado).strip().lower()
	if numerico(resultado) or '<' in resultado: #Adiciona os exames numéricos na lista numericos e os binários na lista binarios
		numericos.append(exame)
	elif resultado_l in ['positivo', 'negativo', 'reagente', 'ausente', 'presente'] or 'nao' in resultado_l or 'não' in resultado_l:
		binarios.append(exame)

#Cria o dicionário de todos os atendimentos
atendimentos = {}

for chunk in pd.read_csv('HC_EXAMES_1.csv', sep='|', usecols=['ID_aTENDIMENTO', 'DE_EXAME', 'DE_RESULTADO'], 
                         chunksize=10000, on_bad_lines='skip', engine='python'):

    for _, row in chunk.iterrows():
        atendimento = row['ID_aTENDIMENTO']
        exame = row['DE_EXAME']
        resultado = row['DE_RESULTADO']

        # Ignora dados ausentes
        if pd.isna(resultado) or pd.isna(exame) or pd.isna(atendimento):
            continue

        # Só considera exames relevantes
        if exame not in binarios and exame not in numericos:
            continue

        # Cria o dicionário para esse atendimento se ainda não existir
        if atendimento not in atendimentos:
            atendimentos[atendimento] = {}

        # Não sobrescreve se já houver o exame
        if exame in atendimentos[atendimento]:
            continue

        # Converte o resultado
        if exame in numericos:
            try:
                valor = float(str(resultado).replace(',', '.'))
            except ValueError:
                valor = 0  # valor padrão se não conseguir converter
        else:
            resultado_l = str(resultado).strip().lower()
            if resultado_l in ['positivo', 'reagente', 'presente']:
                valor = 1
            elif resultado_l in ['negativo', 'ausente'] or 'nao' in resultado_l or 'não' in resultado_l:
                valor = 0
            else:
                valor = 0.5  # valor indefinido para binário

        # Salva o valor
        atendimentos[atendimento][exame] = valor
        
#Remove os atendimentos que não realizaram o exame de IgG
exame_alvo = 'COVID-19 - PESQUISA DE ANTICORPOS IgG'
remover_ids = [k for k, v in atendimentos.items() if exame_alvo not in v]
for k in remover_ids:
    del atendimentos[k]
        
#Adicionam os valores padrão nos exames não realizados e converte os valores do exame de IgG de numérico para binário
todos_exames = sorted(set(numericos) | set(binarios) - {exame_alvo})
linhas = []

for atendimento_id, exames_dict in atendimentos.items():
    linha = {'ID_ATENDIMENTO': atendimento_id}

    for exame in todos_exames:
        valor = exames_dict.get(exame, 0 if exame in numericos else 0.5)
        linha[exame] = valor

    # Converte IgG para binário e salva em 'ALVO'
    valor_igg = exames_dict.get(exame_alvo, 0)
    linha['ALVO'] = int(valor_igg > 1.0)

    linhas.append(linha)
 
df_final = pd.DataFrame(linhas)
    
#Cria coluna IDADE
df_pacientes = pd.read_csv('HC_PACIENTES_1.csv', sep='|', on_bad_lines='skip', engine='python')

ano_atual = datetime.now().year 
df_pacientes['IDADE'] = ano_atual - pd.to_numeric(df_pacientes['AA_NASCIMENTO'], errors='coerce')
df_pacientes['IDADE'] = df_pacientes['IDADE'].fillna(df_pacientes['IDADE'].mean())

df_pacientes['IC_SEXO'] = df_pacientes['IC_SEXO'].map({'M': 0, 'F': 1})

# Adiciona ID_PACIENTE a cada linha
atendimento_paciente = {}

for chunk in pd.read_csv('HC_EXAMES_1.csv', sep='|', usecols=['ID_aTENDIMENTO', 'ID_PACIENTE'], 
                         chunksize=10000, on_bad_lines='skip', engine='python'):

    for _, row in chunk.iterrows():
        atendimento = row['ID_aTENDIMENTO']
        paciente = row['ID_PACIENTE']

        if pd.isna(atendimento) or pd.isna(paciente):
            continue

        if atendimento not in atendimento_paciente:
            atendimento_paciente[atendimento] = paciente

df_final['ID_PACIENTE'] = df_final['ID_ATENDIMENTO'].map(atendimento_paciente)

# Realiza merge entre df_final e df_pacientes por meio de ID_PACIENTE e pega apenas os dados relevantes
df_final = pd.merge(df_final, df_pacientes[['ID_PACIENTE', 'IC_SEXO', 'IDADE']], on='ID_PACIENTE', how='left')

#Escreve o resultado no arquivo
df_final.to_csv('tabela_tratada.csv', index=False)





