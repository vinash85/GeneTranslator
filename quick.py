import pandas as pd
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.stats import pearsonr



df_gene = pd.read_csv('/mnt/data/macaulay/datas/gene_embeddings.csv')
df_gene.drop(columns=['Unnamed: 0'], inplace=True)
df_crispr = pd.read_csv('/mnt/data/macaulay/datas/CRISPRGeneEffect.csv').rename(columns={"ModelID": 'Cell_line'}).set_index('Cell_line')
df_crispr = df_crispr.fillna(df_crispr.mean())
df_crispr.reset_index(inplace=True)
cleaned_headers = [col.split(" ")[0] for col in df_crispr.columns]
df_crispr.columns = cleaned_headers

for i, j in enumerate(range(len(df_gene))):
    df_sgene = df_gene.iloc[j:j+1]
    name = df_sgene.iloc[0,0]
    # print(name)
    try:
         df_crispr1 = df_crispr[['Cell_line',name]]
         pass
    except KeyError:
        print(f'{name} not found in CRISPR data')
        continue

    
    #for i in range (1):
    df_omic = pd.read_csv(f'/mnt/data/macaulay/datas/OmicExpression_embeddings/OmicExpression_embeddings_3.csv')
    df_omic = df_omic[df_omic['Cell_line'].isin(df_crispr1['Cell_line'])]
    df_crispr2 = df_crispr1[df_crispr1['Cell_line'].isin(df_omic['Cell_line'])]
    df2_repeated = pd.concat([df_sgene]*df_omic.shape[0], ignore_index=True)
    df2_repeated = df2_repeated.iloc[:, 1:]
    # Concatenate df1 and repeated df2 along the horizontal axis
    result = pd.concat([df_omic, df2_repeated], axis=1)
    merged_df = pd.merge(result, df_crispr2, left_on='Cell_line', right_on='Cell_line', how='inner')

    X = merged_df.drop(columns=['Cell_line', name])
    y = merged_df[name]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # randomized_columns_order = X_train.columns.to_list()
    # np.random.shuffle(randomized_columns_order)

    # X_train = X_train[randomized_columns_order]
    # X_test = X_test[randomized_columns_order]


    # Training an SVR model with default settings
    svr_model = SVR()
    svr_model.fit(X_train, y_train)

    # Predicting on the test set
    y_pred = svr_model.predict(X_test)

    # Calculating MSE for the predictions
    mse = mean_squared_error(y_test, y_pred)

    #calculate r2 score
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred)

    #calculate pearson correlation

    corr, _ = pearsonr(y_test, y_pred)
    
    print(f'{i},{name}, MSE: {mse}, r2: {r2}, pearson correlation: {corr}')
    with open('quick.txt', 'a') as f:
        if i == 0:
            f.write('index,Gene,MSE,r2,pearson correlation\n')
            f.write(f'{i},{name},{mse},{r2},{corr}\n')
        else:
            f.write(f'{i},{name},{mse},{r2},{corr}\n')
            

