import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import joblib

# === Dados ===
df = pd.read_csv('tabela_tratada.csv')

exame_alvo = 'ALVO'
X = df.drop(columns=[exame_alvo]) 
y = df[exame_alvo]

colunas_para_remover = ['ID_ATENDIMENTO', 'ID_PACIENTE']
X = X.drop(columns=[col for col in colunas_para_remover if col in X.columns])

X = X.loc[:, X.nunique() > 1]  # Remove colunas constantes
colunas_validas = X.columns  # Salva as colunas após remover as constantes
joblib.dump(list(colunas_validas), 'colunas_validas.joblib')

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

n = 50  
selector = SelectKBest(f_classif, k=n)
X = selector.fit_transform(X, y)
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(selector, 'selector.joblib')

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

input_size = X.shape[1]
hidden1_size = 128
hidden2_size = 64
hidden3_size = 128

# === Modelo ===
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

# === K-Fold Cross Validation ===
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
acuracias = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_tensor)):
    print(f"\n===== Fold {fold + 1} =====")

    X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
    y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    modelo = RedeNeural(input_size, hidden1_size, hidden2_size, hidden3_size)
    criterio = nn.BCELoss()
    otimizador = optim.Adam(modelo.parameters(), lr=0.001)

    for epoca in range(30):
        modelo.train()
        perda_media = 0
        for X_batch, y_batch in train_loader:
            otimizador.zero_grad()
            saida = modelo(X_batch)
            perda = criterio(saida, y_batch)
            perda_media += perda
            perda.backward()
            otimizador.step()
            

    # Avaliação
    modelo.eval()
    with torch.no_grad():
        saida_val = modelo(X_val)
        pred_val = (saida_val > 0.5).float()
        y_true = y_val.cpu().numpy()
        y_pred = pred_val.cpu().numpy()

        matriz = confusion_matrix(y_true, y_pred)
        print(f"\nMatriz de Confusão - Fold {fold + 1}:\n{matriz}")

torch.save(modelo.state_dict(), 'modelo_igg.pth')
