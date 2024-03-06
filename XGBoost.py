import numpy as np
import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.model_selection import cross_val_score as cvs
from xgboost import XGBRegressor

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
            df[col].fillna(df[col].median(), inplace = True)
    print(f'Missing values after imputation: {sum(df.isnull().sum())}')
    return df

df = impute_missing_by_median(df)

train = df[df['is_train'] == True]
test = df[df['is_train'] == False]
train.drop('is_train', axis = 1, inplace=True)
test.drop('is_train', axis = 1, inplace=True)
test.reset_index(drop = True, inplace = True)

test.drop(targets_names, axis = 1, inplace = True)
targets = train[targets_names].copy()
train.drop(targets_names, axis = 1, inplace = True)

# additional_columns = ['Id', 'industry', 'fullTimeEmployees', 'auditRisk', 'boardRisk', 'compensationRisk', 'shareHolderRightsRisk', 'overallRisk', 'trailingPE', 'forwardPE', 'floatShares', 'sharesOutstanding', 'trailingEps', 'forwardEps', 'targetHighPrice', 'targetLowPrice', 'targetMeanPrice', 'targetMedianPrice', 'recommendationMean', 'numberOfAnalystOpinions', 'totalCash', 'totalCashPerShare', 'ebitda', 'totalDebt', 'totalRevenue', 'revenuePerShare', 'freeCashflow', 'operatingCashflow', 'revenueGrowth', 'financialCurrency']
# additional_columns = ['Q6_TOTAL_LIABILITIES_AND_EQUITY']

# additional_columns = ['Id']

additional_columns = []
for i in range(1,11):
    additional_columns.append(f"Q{i}_TOTAL_LIABILITIES_AND_EQUITY")
    additional_columns.append(f"Q{i}_DEPRECIATION_AND_AMORTIZATION")
#     additional_columns.append(f"Q{i}_fiscal_year_end")
#     additional_columns.append(f"Q{i}_TOTAL_CURRENT_ASSETS")
#     additional_columns.append(f"Q{i}_TOTAL_NONCURRENT_ASSETS")
#     additional_columns.append(f"Q{i}_TOTAL_CURRENT_LIABILITIES")
#     additional_columns.append(f"Q{i}_TOTAL_NONCURRENT_LIABILITIES")

test.drop(additional_columns, axis=1, inplace=True)
train.drop(additional_columns, axis=1, inplace=True)

model = XGBRegressor(eta= 0.01, max_depth=10, n_estimators=500, device='cuda')

print('Cross validation R2 scores for each target:\n')
cross_val_score_results = {}
for target in targets:
    score = np.round(np.mean(cvs(model, train, targets[target], cv=3, scoring='r2')),2)
    cross_val_score_results[target] = score
    print(f'{target} -> {score}')
print(f'\nMean R2 score across all targets: {np.mean(list(cross_val_score_results.values()))}')

preds = {}
for target in targets:
    model.fit(train, targets[target])
    pred = model.predict(test)
    preds[target] = pred
    print(f'Finished train/predict for: {target}')

sub = pd.read_csv('sample_submission.csv')

# sanity check
sub.Id == test.Id

for target in preds:
    sub[target] = preds[target]
sub.head()

sub.to_csv('submission_2_ML.csv', index = False)
print("Finished!")
