#a.py
"""

Goal
Test the model on the saved models, and calculate the metrics for each gene concatinated with the cell line embeddings

it calculate the correlation by each seen gene
"""
import torch 
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import torch.nn.functional as F
from functions import FullConnectedBlock, NeuralNetwork
pd.options.mode.chained_assignment = None





# Initialize lists to store metrics


avg_test_losses = []
mae_values = []
rmse_values = []
r2_values = []
correlation_coefficients = []
p_values = []



# Loop through the saved models
# for epoch in range(num_epochs):

target_epochs = [2]

# Loop through the target epochs
for epoch in target_epochs:
    #model_epoch = epoch
    Y_data_count = 0

    model_path = f'/mnt/data/macaulay/datas/genepedia/crispr_fc1_model_state_epoch_{epoch}.pth'
    if os.path.exists(model_path):
        

        data1_test = pd.read_csv("/mnt/data/macaulay/datas/test_omicExpression_Embeddings.csv") #30 >>
        data2_test = pd.read_csv("/mnt/data/macaulay/datas/training_gene_embeddings.csv") #17000 >>
        df_Y_test = pd.read_csv('/mnt/data/macaulay/datas/A_test_gene__Y_crispr.csv') #>>
        print('Test data loaded successfully')




        batch_size1 = len(data1_test)
        batch_size_Y = len(data1_test)
        batch_size2 = 1
        combined_batches_test = []

        input_dim = data1_test.shape[1] + data2_test.shape[1] - 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Device:', device)
        model = NeuralNetwork(input_dim)

        # If multiple GPUs
        if torch.cuda.device_count() >= 4:
            print("Using 4 GPUs!")
            model = nn.DataParallel(model, device_ids=list(range(4)))

        model.to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set the model to evaluation mode

        # Define the loss function as Mean Squared Error
        loss_fn = nn.MSELoss()

        

        for i in range(0, len(data2_test), batch_size2):  # Loop through data2_test
            batch_data2_test = data2_test.iloc[i:i+batch_size2]
            for j in range(0, len(data1_test), batch_size1):  # Loop through data1_test
                batch_data1_test = data1_test.iloc[j:j+batch_size1]
                batch_data1_test['key'] = 1
                batch_data2_test['key'] = 1
                batch_data1_test = batch_data1_test[list(data1_test.columns[0:]) + ['key']]
                batch_data2_test = batch_data2_test[list(data2_test.columns[1:]) + ['key']]
                combined_batch_test = pd.merge(batch_data1_test, batch_data2_test, on='key').drop(columns=['key'])
                combined_batches_test.append(combined_batch_test)

                X_test = pd.concat(combined_batches_test)
                combined_batches_test = []
                Y_test = df_Y_test.iloc[Y_data_count:(batch_size_Y * (i+1))]
                Y_data_count += batch_size_Y

                
                X_test1 = torch.tensor(X_test.values, dtype=torch.float32)
                Y_test1 = torch.tensor(Y_test.values.reshape(-1, 1), dtype=torch.float32)
                test_data = TensorDataset(X_test1, Y_test1)
                test_dataloader = DataLoader(test_data, batch_size=128, shuffle=True)

                #print('Test data preprocessed successfully')
                ##################### The first 1 Batch which forms 100 rows after concat, are preprocessed and fed to the neural network under the same loop #####################

                # Evaluate the model on the test data
                test_loss = 0.0
                actual_outputs = []
                predicted_outputs = []
                with torch.no_grad():  # Disable gradient calculation
                    for inputs, targets in test_dataloader:
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        outputs = model(inputs)
                        loss = loss_fn(outputs, targets)
                        test_loss += loss.item()
                        actual_outputs.extend(targets.cpu().numpy().flatten().tolist())
                        predicted_outputs.extend(outputs.cpu().numpy().flatten().tolist())

                avg_test_loss = test_loss / len(test_dataloader)
                avg_test_losses.append(avg_test_loss)

                # Calculate Pearson correlation coefficient
                correlation_coefficient, p_value = pearsonr(actual_outputs, predicted_outputs)
                correlation_coefficients.append(correlation_coefficient)
                p_values.append(p_value)  # Append the p-value

                mae = mean_absolute_error(actual_outputs, predicted_outputs)
                rmse = np.sqrt(mean_squared_error(actual_outputs, predicted_outputs))
                r2 = r2_score(actual_outputs, predicted_outputs)

                # Append metrics to their respective lists
                mae_values.append(mae)
                rmse_values.append(rmse)
                r2_values.append(r2)

                print(f'Epoch {epoch} : gene {i+1}/{target_epochs}: Avg. test loss = {avg_test_loss:.4f}')
                print(f'Epoch {epoch} : gene {i+1}/{target_epochs}: Correlation Coefficient = {correlation_coefficient:.4f}')
                print(f'Epoch {epoch} : gene {i+1}/{target_epochs}: Mean Absolute Error = {mae:.4f}')
                print(f'Epoch {epoch} : gene {i+1}/{target_epochs}: Root Mean Square Error = {rmse:.4f}')
                print(f'Epoch {epoch} : gene {i+1}/{target_epochs}: R2 Score = {r2:.4f}')


    else:
        print(f'Model for epoch {epoch} not found!')


# Create a list to store the full epoch labels
epochs = []

# Loop through the target epochs
for major in target_epochs:
    for minor in range(1, len(data2_test) + 1):
        epoch = f'{major}.{minor:02}'  # Format as a string with two decimal places
        epochs.append(float(epoch))

# Create a DataFrame to hold the metrics
metrics_df = pd.DataFrame({
    'Epoch': epochs,
    'Correlation_Coefficient': correlation_coefficients,
    'P_Value': p_values,  # Include the p-values
    'Test_Loss': avg_test_losses,
    'MAE': mae_values,
    'RMSE': rmse_values,
    'R2_Score': r2_values
})

# Write the DataFrame to a CSV file
metrics_path = f'/mnt/data/macaulay/datas/genepedia/A_seen_genewise_correlation_metrics_summary_crispr_{target_epochs[0]}.csv'
metrics_df.to_csv(metrics_path, index=False)
print(f'Metrics saved to {metrics_path}')

























import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
from functions import FullConnectedBlock, NeuralNetwork
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
pd.options.mode.chained_assignment = None
import numpy as np




def initialize_environment(data1_path, data2_path, df_Y_path, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    print(f'Number of available GPUs: {torch.cuda.device_count()}')

    data1 = pd.read_csv(data1_path)
    data2 = pd.read_csv(data2_path)
    df_Y = pd.read_csv(df_Y_path)
    print('Training data loaded successfully')

    loss_fn = nn.MSELoss()
    input_dim = data1.shape[1] + data2.shape[1] - 1
    model = NeuralNetwork(input_dim)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    if torch.cuda.device_count() >= 4:
        print("Using 4 GPUs!")
        model = nn.DataParallel(model, device_ids=list(range(4)))
    print('Model initialized successfully')
    
    
    return device, data1, data2, df_Y, model, optimizer, loss_fn

def cartesian_product(data1, data2):
    data1['key'] = 1
    data2['key'] = 1
    combined_data = pd.merge(data1, data2, on='key').drop(columns=['key'])
    print(len(combined_data))
    
    return combined_data

def cartesian_product_generator(data1, data2, df_Y, batch_size1):
    for i in range(0, len(data1), batch_size1):
        start_idx = i * len(data2)
        end_idx = (i + batch_size1) * len(data2)
        batch_data1 = data1.iloc[i:i + batch_size1]
        
        batch_X = cartesian_product(batch_data1, data2)
        batch_Y = df_Y.iloc[start_idx:end_idx]
        yield batch_X, batch_Y

def load_model(model, epoch, model_save_path):
    model_path = os.path.join(model_save_path, f'crispr_fc1_model_state_epoch_{epoch-1}.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f'Model {epoch - 1} loaded successfully for epoch {epoch}')
    else:
        print('No saved model found. Training from scratch.')
    return model


def load_test_data(data1_test_path_A, data2_test_path_A, df_Y_test_path_A,
                    data1_test_path_B, data2_test_path_B, df_Y_test_path_B,
                      data1_test_path_C, data2_test_path_C, df_Y_test_path_C):
    
    data1_test_A = pd.read_csv(data1_test_path_A)
    data2_test_A = pd.read_csv(data2_test_path_A)
    df_Y_test_A = pd.read_csv(df_Y_test_path_A)
    data1_test_B = pd.read_csv(data1_test_path_B)
    data2_test_B = pd.read_csv(data2_test_path_B)
    df_Y_test_B = pd.read_csv(df_Y_test_path_B)
    data1_test_C = pd.read_csv(data1_test_path_C)
    data2_test_C = pd.read_csv(data2_test_path_C)
    df_Y_test_C = pd.read_csv(df_Y_test_path_C)
    print('Test data loaded successfully')
    return data1_test_A, data2_test_A, df_Y_test_A, data1_test_B, data2_test_B, df_Y_test_B, data1_test_C, data2_test_C, df_Y_test_C


# Evaluation Function
def evaluate_model_on_test_data(model, data1_test, data2_test, df_Y_test, epoch, loss_fn, device, test_batch_size=128):
    model.eval()
    avg_test_losses = []
    mae_values = []
    rmse_values = []
    r2_values = []
    correlation_coefficients = []
    p_values = []
    gene_batches = 1
    for batch_X, batch_Y in cartesian_product_generator(data1_test, data2_test, df_Y_test, gene_batches):
        X_test = torch.tensor(batch_X.values, dtype=torch.float32).to(device)
        Y_test = torch.tensor(batch_Y.values.reshape(-1, 1), dtype=torch.float32).to(device)
        test_data = TensorDataset(X_test, Y_test)
        test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True)

        test_loss = 0.0
        actual_outputs = []
        predicted_outputs = []
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                test_loss += loss.item()
                actual_outputs.extend(targets.cpu().numpy().flatten().tolist())
                predicted_outputs.extend(outputs.cpu().numpy().flatten().tolist())

        avg_test_loss = test_loss / len(test_dataloader)
        avg_test_losses.append(avg_test_loss)
        correlation_coefficient, p_value = pearsonr(actual_outputs, predicted_outputs)
        correlation_coefficients.append(correlation_coefficient)
        p_values.append(p_value)
        mae = mean_absolute_error(actual_outputs, predicted_outputs)
        rmse = np.sqrt(mean_squared_error(actual_outputs, predicted_outputs))
        r2 = r2_score(actual_outputs, predicted_outputs)
        mae_values.append(mae)
        rmse_values.append(rmse)
        r2_values.append(r2)

    epochs = []
    for minor in range(1, len(data1_test) + 1):
        epoched = f'{epoch}.{minor:02}'  
        epochs.append(float(epoched))

    metrics_df = pd.DataFrame({
        'Epoch': epochs,
        'Correlation_Coefficient': correlation_coefficients,
        'P_Value': p_values,
        'Test_Loss': avg_test_losses,
        'MAE': mae_values,
        'RMSE': rmse_values,
        'R2_Score': r2_values
    })
    return metrics_df



# Main Training Loop
def main_training_loop(data1_path, data2_path, df_Y_path, 
                               data1_test_path_A, data2_test_path_A, df_Y_test_path_A, 
                               data1_test_path_B, data2_test_path_B, df_Y_test_path_B,
                               data1_test_path_C, data2_test_path_C, df_Y_test_path_C,
                               batch_size1, learning_rate, num_epochs, model_save_path, test_batch_size=128):
    
# Evaluate the model on test data after each epoch

    data1_test_A, data2_test_A, df_Y_test_A, data1_test_B, data2_test_B, df_Y_test_B, data1_test_C, data2_test_C, df_Y_test_C = load_test_data(data1_test_path_A, data2_test_path_A, df_Y_test_path_A,
                                                                                                                                                data1_test_path_B, data2_test_path_B, df_Y_test_path_B,
                                                                                                                                                  data1_test_path_C, data2_test_path_C, df_Y_test_path_C)
    target_epochs = [2]

    
    for epoch in target_epochs:


        metrics_df_A = evaluate_model_on_test_data(model, data1_test_A, data2_test_A, df_Y_test_A, epoch, loss_fn, device, test_batch_size)
        metrics_path_A = os.path.join(model_save_path, f"A_metrics_epoch_{epoch}.csv")
        metrics_df_A.to_csv(metrics_path_A, index=False)
        print(f'Metrics for epoch {epoch} saved to {metrics_path_A}')

        metrics_df_B = evaluate_model_on_test_data(model, data1_test_B, data2_test_B, df_Y_test_B, epoch, loss_fn, device, test_batch_size)
        metrics_path_B = os.path.join(model_save_path, f"B_metrics_epoch_{epoch}.csv")
        metrics_df_B.to_csv(metrics_path_B, index=False)
        print(f'Metrics for epoch {epoch} saved to {metrics_path_B}')

        metrics_df_C = evaluate_model_on_test_data(model, data1_test_C, data2_test_C, df_Y_test_C, epoch, loss_fn, device, test_batch_size)
        metrics_path_C = os.path.join(model_save_path, f"C_metrics_epoch_{epoch}.csv")
        metrics_df_C.to_csv(metrics_path_C, index=False)
        print(f'Metrics for epoch {epoch} saved to {metrics_path_C}')







main_training_loop(data1_path="/mnt/data/macaulay/datas/training_gene_embeddings.csv",
                   data2_path="/mnt/data/macaulay/datas/training_omicExpression_Embeddings.csv",
                   df_Y_path='/mnt/data/macaulay/datas/training_crispr.csv',
                   data1_test_path_A="/mnt/data/macaulay/datas/training_gene_embeddings.csv",
                   data2_test_path_A="/mnt/data/macaulay/datas/test_omicExpression_Embeddings.csv",
                   df_Y_test_path_A='/mnt/data/macaulay/datas/A_test_gene__Y_crispr.csv',
                   data1_test_path_B="/mnt/data/macaulay/datas/test_gene_embeddings.csv",
                   data2_test_path_B="/mnt/data/macaulay/datas/training_omicExpression_Embeddings.csv",
                   df_Y_test_path_B='/mnt/data/macaulay/datas/B_test_gene__Y_crispr.csv',
                   data1_test_path_C="/mnt/data/macaulay/datas/test_gene_embeddings.csv",
                   data2_test_path_C="/mnt/data/macaulay/datas/test_omicExpression_Embeddings.csv",
                   df_Y_test_path_C='/mnt/data/macaulay/datas/C_test_gene__Y_crispr.csv',
                   batch_size1=500,learning_rate=0.001,num_epochs=10,
                   model_save_path='/mnt/data/macaulay/datas/genepedia/',test_batch_size=128)

   
