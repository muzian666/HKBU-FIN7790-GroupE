import numpy as np
import pandas as pd
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

targets_names = [col for col in train if col not in test]
print(f'Target columns: {targets_names}')

# make an easy col to split back the train and test sets
train['is_train'] = True
test['is_train'] = False
df = pd.concat([train, test], axis=0, ignore_index=True)


def binarize(df, col):
    uniques = list(df[col].dropna().unique())
    df[col] = [1 if val == uniques[0] else 0 for val in df[col]]
    return df


def one_hot_encode(df, col):
    dummies = pd.get_dummies(df[col], prefix=col).astype(int)
    df.drop(col, axis=1, inplace=True)
    df = pd.concat([df, dummies], axis=1)
    return df


def factorize(df, col):
    mapping = {name: ix for ix, name in enumerate(df[col].unique())}
    df[col] = df[col].map(mapping)
    return df


df = binarize(df, 'financialCurrency')
print(df['financialCurrency'].head())

print(df.shape)
df = one_hot_encode(df, 'sector')
df = one_hot_encode(df, 'recommendationKey')
print(df.shape)

df = factorize(df, 'industry')
print(df.dtypes.value_counts())

def impute_missing_by_median(df):
    print(f'Missing values before imputation: {sum(df.isnull().sum())}')
    # first replace inf and -inf with nan
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # then impute nan by median
    for col in df:
        if np.any(df[col].isnull()):
            df[col].fillna(df[col].median(), inplace=True)
    print(f'Missing values after imputation: {sum(df.isnull().sum())}')
    return df

df = impute_missing_by_median(df)

train = df[df['is_train'] == True]
test = df[df['is_train'] == False]
train.drop('is_train', axis=1, inplace=True)
test.drop('is_train', axis=1, inplace=True)
test.reset_index(drop=True, inplace=True)

test.drop(targets_names, axis=1, inplace=True)
targets = train[targets_names].copy()
train.drop(targets_names, axis=1, inplace=True)

additional_columns = ['Id']
# for i in range(1, 11):
#     additional_columns.append(f"Q{i}_TOTAL_LIABILITIES_AND_EQUITY")
#     additional_columns.append(f"Q{i}_fiscal_year_end")
#     additional_columns.append(f"Q{i}_TOTAL_CURRENT_ASSETS")
#     additional_columns.append(f"Q{i}_TOTAL_NONCURRENT_ASSETS")
#     additional_columns.append(f"Q{i}_TOTAL_CURRENT_LIABILITIES")
#     additional_columns.append(f"Q{i}_TOTAL_NONCURRENT_LIABILITIES")

test.drop(additional_columns, axis=1, inplace=True)
train.drop(additional_columns, axis=1, inplace=True)

# Deep Learning Model
class RegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(64, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(128, 256)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(256, 512)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.2)

        self.fc5 = nn.Linear(512, 1024)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(0.2)

        self.fc6 = nn.Linear(1024, 1024)
        self.relu6 = nn.ReLU()
        self.dropout6 = nn.Dropout(0.2)

        self.fc7 = nn.Linear(1024, 512)
        self.relu7 = nn.ReLU()
        self.dropout7 = nn.Dropout(0.2)

        self.fc8 = nn.Linear(512, 256)
        self.relu8 = nn.ReLU()
        self.dropout8 = nn.Dropout(0.2)

        self.fc9 = nn.Linear(256, 128)
        self.relu9 = nn.ReLU()
        self.dropout9 = nn.Dropout(0.2)

        self.fc10 = nn.Linear(128, 64)
        self.relu10 = nn.ReLU()
        self.dropout10 = nn.Dropout(0.2)

        self.fc11 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.dropout3(self.relu3(self.fc3(x)))
        x = self.dropout4(self.relu4(self.fc4(x)))
        x = self.dropout5(self.relu5(self.fc5(x)))
        x = self.dropout6(self.relu6(self.fc6(x)))
        x = self.dropout7(self.relu7(self.fc7(x)))
        x = self.dropout8(self.relu8(self.fc8(x)))
        x = self.dropout9(self.relu9(self.fc9(x)))
        x = self.dropout10(self.relu10(self.fc10(x)))
        x = self.fc11(x)
        return x

# Prepare data for PyTorch
scaler = StandardScaler()
X_train = scaler.fit_transform(train)
X_test = scaler.transform(test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

preds = {}
r2_scores = {}
for target in targets_names:
    print(f'Training for target: {target}')
    y = targets[target].values
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = RegressionModel(X_train.shape[1], 1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    losses = []

    # Training loop
    for epoch in tqdm(range(300)):
        epoch_loss = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(dataloader)
        losses.append(epoch_loss)

    # Plotting the loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.savefig(f'ResultIMG/{target}LossCurve.png')
    plt.show()

    # Prediction and R2 score calculation
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        preds[target] = model(X_test_tensor).cpu().numpy().squeeze()
        r2_scores[target] = r2_score(y, model(X_train_tensor).cpu().numpy().squeeze())
        print(f'R2 score for {target}: {r2_scores[target]}')


# print(model)

sub = pd.read_csv('sample_submission.csv')
for target in preds:
    sub[target] = preds[target]
sub.head()

sub.to_csv('submission_11_DL.csv', index=False)
print("Finished!")
