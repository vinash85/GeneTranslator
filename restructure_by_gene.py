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
    # print(len(combined_data))
    
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


# Main Training Loop
def main_training_loop(data1_path, data2_path, df_Y_path, 
                               data1_test_path_A, data2_test_path_A, df_Y_test_path_A, 
                               data1_test_path_B, data2_test_path_B, df_Y_test_path_B,
                               data1_test_path_C, data2_test_path_C, df_Y_test_path_C,
                               batch_size1, learning_rate, num_epochs, model_save_path, test_batch_size=128):
    

    device, data1, data2, df_Y, model, optimizer, loss_fn = initialize_environment(data1_path, data2_path, df_Y_path, learning_rate)
    # data1 = data1.iloc[:300]

    # display(data1)
    # display(data2)
    # display(df_Y)



    # import time

    # start_epoch = 1
    # end_epoch = 20
    from sklearn.model_selection import train_test_split
    from scipy.stats import pearsonr
    # print(len(data1))
    saved_models = os.listdir(model_save_path)
    epochs = [int(file.split('_')[-1].split('.')[0]) for file in saved_models if 'crispr_fc1_model_state_epoch_' in file]
    last_epoch = max(epochs) if epochs else 0
    start_epoch = last_epoch + 1
    end_epoch = start_epoch + num_epochs


    gene_predictions = {}
    gene_actuals = {}



    for epoch in range(start_epoch, end_epoch):
        max_index = (len(data1) // batch_size1) * batch_size1
        # print(max_index)
        # # time.sleep(10000)
        max_index = 50 

        epoch_train_correlations = []
        epoch_train_losses = []
        epoch_test_correlations = []
        epoch_test_losses = []
        test_counter = 0
        for j in range(0, max_index, batch_size1):
            batched_data1 = data1.iloc[j:j+batch_size1]
            model.train()
            if j == 0:
                model = load_model(model, epoch, model_save_path)

            for batch_X, batch_Y in cartesian_product_generator(batched_data1, data2, df_Y, batch_size1):
                
                # Concatenate X and Y DataFrames along the columns axis
                combined = pd.concat([batch_X.reset_index(drop=True), batch_Y.reset_index(drop=True)], axis=1)            

                # Shuffle data
                combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
                
                


                # Split shuffled data back into X and Y
                shuffled_X = combined.iloc[:, :-batch_Y.shape[1]]
                shuffled_Y = combined.iloc[:, -batch_Y.shape[1]:]

                # Then use train_test_split to split the shuffled data into training and test sets
                X_train, X_test, Y_train, Y_test = train_test_split(shuffled_X, shuffled_Y, test_size=0.05, shuffle=False)
                # print(X_train)
                # print(X_train.shape)
                # print(X_train['Gene'].unique())
                # if test_counter // 50 == 0:
                #     print(f'Gene {j}')
                if epoch == 1:
                    file_name_X = f'X_test_{test_counter}.csv'
                    file_path_X = os.path.join(model_save_path, file_name_X)
                    file_name_Y = f'Y_test_{test_counter}.csv'
                    file_path_Y = os.path.join(model_save_path, file_name_Y)
                    X_test.to_csv(file_path_X, index=False)
                    Y_test.to_csv(file_path_Y, index=False)
                    X_train = X_train.drop(columns=["Gene"])
                    
                    test_counter += 1
                else:
                    file_name_X = f'X_test_{test_counter}.csv'
                    file_path_X = os.path.join(model_save_path, file_name_X)
                    X_test = pd.read_csv(file_path_X)
                    file_name_Y = f'Y_test_{test_counter}.csv'
                    file_path_Y = os.path.join(model_save_path, file_name_Y)
                    Y_test = pd.read_csv(file_path_Y)
                    X_train = X_train.drop(columns=["Gene"])
                    
                    test_counter += 1


                # import sys
                # sys.exit('stop')
                # Convert training data to tensor and create dataloader
                X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
                Y_train_tensor = torch.tensor(Y_train.values.reshape(-1, 1), dtype=torch.float32).to(device)
                train_data = TensorDataset(X_train_tensor, Y_train_tensor)
                train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)
                
                train_loss = 0.0
                train_correlations_b = []
                for inputs, targets in train_dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    train_predictions_np = outputs.detach().cpu().numpy()
                    Y_train_np = targets.cpu().numpy()
                    train_correlation, _ = pearsonr(train_predictions_np.squeeze(), Y_train_np.squeeze())
                    train_correlations_b.append(train_correlation)
                    # train_lossess.append(loss.item())
                epoch_train_losses.append(train_loss / len(train_dataloader))
                epoch_train_correlations.append(np.mean(train_correlations_b))



                # Evaluate on test data
                model.eval()
                with torch.no_grad():
                    # Convert test data to tensor and evaluate
                    X_test_tensor = torch.tensor(X_test.drop(columns=["Gene"]).values, dtype=torch.float32).to(device)
                    Y_test_tensor = torch.tensor(Y_test.values.reshape(-1, 1), dtype=torch.float32).to(device)
                    predictions = model(X_test_tensor)
                    test_loss = loss_fn(predictions, Y_test_tensor)
                    
                    # Convert tensors to numpy arrays for correlation computation
                    predictions_np = predictions.cpu().numpy()
                    Y_test_np = Y_test_tensor.cpu().numpy()
                    
                    correlation, _ = pearsonr(predictions_np.squeeze(), Y_test_np.squeeze())
                    epoch_test_correlations.append(correlation)
                    epoch_test_losses.append(test_loss.item())

                    # Aggregate predictions and actual values for each gene
                    for gene in X_test['Gene'].unique():
                        gene_mask = (X_test['Gene'] == gene)
                        X_test_gene = X_test[gene_mask]
                        Y_test_gene = Y_test[gene_mask]

                        X_test_tensor = torch.tensor(X_test_gene.drop(columns=["Gene"]).values, dtype=torch.float32).to(device)
                        Y_test_tensor = torch.tensor(Y_test_gene.values.reshape(-1, 1), dtype=torch.float32).to(device)

                        predictions_gene = model(X_test_tensor)

                        predictions_gene_np = predictions_gene.cpu().numpy().squeeze()
                        Y_test_gene_np = Y_test_tensor.cpu().numpy().squeeze()
                        gene_predictions[gene] = gene_predictions.get(gene, []) + predictions_gene_np.tolist()
                        gene_actuals[gene] = gene_actuals.get(gene, []) + Y_test_gene_np.tolist()

                # Calculate Pearson correlation for each gene
                gene_correlations = {}
                for gene in gene_predictions.keys():
                    correlation, _ = pearsonr(gene_predictions[gene], gene_actuals[gene])
                    gene_correlations[gene] = correlation

        # Print or save gene correlations
        for gene, corr in gene_correlations.items():
            print(f"Gene {gene}: Pearson Correlation = {corr:.4f}")

            with open(os.path.join(model_save_path, f'gene_correlations.txt'), 'a') as f:
                if epoch == 1:
                    f.write('Epoch,gene,correlation\n')
                f.write(f'Epoch{epoch},{gene},{corr:.4f}\n')
                   

        # Save model state and metrics
        torch.save(model.state_dict(), os.path.join(model_save_path, f'crispr_fc1_model_state_epoch_{epoch}.pth'))
        with open(os.path.join(model_save_path, f'metrics.txt'), 'a') as f:
            if epoch == 1:
                f.write('epoch,train_loss,train_correlation,test_loss,test_correlation\n')
            f.write(f'{epoch},{np.mean(epoch_train_losses):.4f},{np.mean(epoch_train_correlations):.4f},{np.mean(epoch_test_losses):.4f},{np.mean(epoch_test_correlations):.4f}\n')
        print(f'Epoch {epoch} completed')




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
                   batch_size1=1,learning_rate=0.001,num_epochs=6,
                   model_save_path='/home/macaulay/macaulay/GenePedia/datas/',test_batch_size=128)

                #    model_save_path='/mnt/data/macaulay/datas/model_datas/',test_batch_size=128)

# model_save_path='/home/macaulay/macaulay/GenePedia/datas/',test_batch_size=128)