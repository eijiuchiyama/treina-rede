import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib 

class RedeNeural(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, hidden3_size):
        super(RedeNeural, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, hidden3_size)
        self.out = nn.Linear(hidden3_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.out(x))
        return x

df = pd.read_csv('dados_validar.csv')
print(df)
ids = df['ID_ATENDIMENTO'] if 'ID_ATENDIMENTO' in df.columns else range(len(df))

if 'ID_ATENDIMENTO' in df.columns:
    df = df.drop(columns=['ID_ATENDIMENTO'])
    
colunas_validas = joblib.load('colunas_validas.joblib')
df = df[colunas_validas]

scaler = joblib.load('scaler.joblib')
selector = joblib.load('selector.joblib')

df_scaled = scaler.transform(df)
df_selected = selector.transform(df_scaled)

input_size = df_selected.shape[1]
modelo = RedeNeural(input_size, 128, 64, 128)
modelo.load_state_dict(torch.load('modelo_igg.pth'))
modelo.eval()

X_pred = torch.tensor(df_selected, dtype=torch.float32)
with torch.no_grad():
    saida = modelo(X_pred)
    predicoes = (saida > 0.5).int().flatten().numpy()

resultado = pd.DataFrame({
    'ID_ATENDIMENTO': list(ids)[:len(predicoes)],
    'RESULTADO_PREDITO': predicoes
})
resultado.to_csv('predicoes.csv', index=False)
print(resultado.head())
