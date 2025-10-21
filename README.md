# Treina-Rede

## Directories

## PRE (preprocessing) directory

trata_dados.py reads the HC_EXAMES_1.csv and HC_PACIENTES_1.csv files, treats its data, join them in a single file and saves it in the file tabela_tratada.csv with the correct values to be used for the neural network.

The files HC_EXAMES_1.csv and HC_PACIENTES.csv couldn't be pushed to this repository due to their large size. They can be, however, downloaded from https://repositoriodatasharingfapesp.uspdigital.usp.br/handle/item/100

### NN (neural network) directory

treina_rede_neural.py trains a neural network using the tabela_tratada.csv file, that has already been treated. Once executed, it creates the modelo_igg.pth file, that saves the neural network model.

valida_dados.py does the prediction using the data of the patients we want to predict in dados_validar.csv and creates a new file, predictions.csv, with the results.

This work used data obtained from the COVID-19 Data Sharing/BR, available at https://repositoriodatasharingfapesp.uspdigital.usp.br/.
