import pandas as pd
from sklearn.preprocessing import OneHotEncoder
cat_cols_essential = [
    'IsSeniorCitizen',
    'HasPartner',
    'HasChild',
    'HasInternetService',
    'HasOnlineSecurityService',
    'HasOnlineBackup',
    'HasDeviceProtection',
    'HasTechSupportAccess',
    'HasOnlineTV',
    'HasMovieSubscription',
    'HasContractPhone',
    'IsBillingPaperless',
    'PaymentMethod'
]
num_cols = [
    'ClientPeriod',
    'MonthlySpending',
    'TotalSpent'
]
target_col = 'Churn'
# Категориальные признаки
cat_cols = [
    'Sex',
    'IsSeniorCitizen',
    'HasPartner',
    'HasChild',
    'HasPhoneService',
    'HasMultiplePhoneNumbers',
    'HasInternetService',
    'HasOnlineSecurityService',
    'HasOnlineBackup',
    'HasDeviceProtection',
    'HasTechSupportAccess',
    'HasOnlineTV',
    'HasMovieSubscription',
    'HasContractPhone',
    'IsBillingPaperless',
    'PaymentMethod'
]

data = pd.read_csv('train.csv')
work_cols = cat_cols_essential + num_cols

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler

def change(x):
    if x == " ":
        x = 800
    else:
        x = float(x)
    return x
data.TotalSpent = data.TotalSpent.apply(change)

data_cat = data[cat_cols_essential]


scaler = StandardScaler()
scaler.set_output(transform="pandas")
data_num_scale = scaler.fit_transform(data[num_cols])


trans = make_column_transformer((OneHotEncoder(drop="if_binary", sparse_output=False), cat_cols_essential), remainder="passthrough")
trans_ed = trans.fit_transform(data_cat)
data_new = pd.DataFrame(trans_ed, columns=trans.get_feature_names_out())
data_new[num_cols] = data_num_scale
data_new[target_col] = data[target_col]

print(type(data_new), data_new.shape, data_new.head())




