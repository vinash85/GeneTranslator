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




# Initialization and Environment Setup
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



def cartesian_product(data1, data2):
    data1 = data1.iloc[:, 1:]
    data1['key'] = 1
    data2['key'] = 1
    combined_data = pd.merge(data1, data2, on='key').drop(columns=['key'])
    return combined_data

def cartesian_product_generator(data1, data2, df_Y, batch_size1):
    for i in range(0, len(data1), batch_size1):
        start_idx = i * len(data2)
        end_idx = (i + batch_size1) * len(data2)
        batch_data1 = data1.iloc[i:i + batch_size1]
        combined_data = cartesian_product(batch_data1, data2)
        batch_Y = df_Y.iloc[start_idx:end_idx]
        yield combined_data, batch_Y

def load_model(model, epoch, model_save_path):
    model_path = os.path.join(model_save_path, f'crispr_fc1_model_state_epoch_{epoch-1}.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f'Model {epoch - 1} loaded successfully for epoch {epoch}')
    else:
        print('No saved model found. Training from scratch.')
    return model

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
    

    device, data1, data2, df_Y, model, optimizer, loss_fn = initialize_environment(data1_path, data2_path, df_Y_path, learning_rate)

        
    saved_models = os.listdir(model_save_path)
    epochs = [int(file.split('_')[-1].split('.')[0]) for file in saved_models if 'crispr_fc1_model_state_epoch_' in file]
    last_epoch = max(epochs) if epochs else 0
    start_epoch = last_epoch + 1
    end_epoch = start_epoch + num_epochs
    
    total_training_losses = []
    for epoch in range(start_epoch, end_epoch):
        training_losses = []
        model.train()
        counter = 0
        model = load_model(model, epoch, model_save_path)
        for batch_X, batch_Y in cartesian_product_generator(data1, data2, df_Y, batch_size1):
            X_train = torch.tensor(batch_X.values, dtype=torch.float32).to(device)
            Y_train = torch.tensor(batch_Y.values.reshape(-1, 1), dtype=torch.float32).to(device)
            train_data = TensorDataset(X_train, Y_train)
            train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)

            train_loss = 0.0
            for inputs, targets in train_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            counter += 1
            avg_train_loss = train_loss / len(train_dataloader)
            print(f"Epoch {epoch}.{counter}: Avg. training loss = {avg_train_loss:.4f}")
            training_losses.append(avg_train_loss)

        # Calculate the mean of training_losses for the current epoch and append to total_training_losses
        mean_epoch_loss = sum(training_losses) / len(training_losses)
        total_training_losses.append(mean_epoch_loss)
        


        torch.save(model.state_dict(), os.path.join(model_save_path, f'crispr_fc1_model_state_epoch_{epoch}.pth'))
        
        # Evaluate the model on test data after each epoch

        data1_test_A, data2_test_A, df_Y_test_A, data1_test_B, data2_test_B, df_Y_test_B, data1_test_C, data2_test_C, df_Y_test_C = load_test_data(data1_test_path_A, data2_test_path_A, df_Y_test_path_A,
                                                                                                                                                data1_test_path_B, data2_test_path_B, df_Y_test_path_B,
                                                                                                                                                  data1_test_path_C, data2_test_path_C, df_Y_test_path_C)



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

    # Create a DataFrame from total_training_losses
    total_train_loss_df = pd.DataFrame(total_training_losses)
    total_train_loss_df.columns = ['Average Training Loss']

    # Save the DataFrame to a CSV file
    total_train_loss_df.to_csv('/mnt/data/macaulay/datas/average_training_losses.csv', index=False)

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
                   batch_size1=500,learning_rate=0.001,num_epochs=20,
                   model_save_path='/mnt/data/macaulay/model_state2/',test_batch_size=128)

    