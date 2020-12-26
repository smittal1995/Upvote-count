from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
import pandas as pd

train = pd.read_csv('train_NIR5Yl1.csv')
train.head()

train.drop(['ID','Username'],axis=1,inplace=True)

le = LabelEncoder()
train['Tag'] = le.fit_transform(train['Tag'])
print(train.head())

X=train.drop('Upvotes',axis=1)
y=train['Upvotes']

ts = 0.24
rs = 205
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=ts, random_state=rs)

print(X_train.head())
print(X_val.head())

est = 30
md = 7
rs = 205
regr = RandomForestRegressor(n_estimators=est ,max_depth=md, random_state=rs, verbose=2)
regr.fit(X_train, y_train)
predictions = regr.predict(X_train)
print('Train R2:',r2_score(y_train, predictions))
print('Train RMSE:',np.sqrt(mean_squared_error(y_train, predictions)))

test = pd.read_csv('test_8i3B3FC.csv')
ID = test['ID']
test.drop(['ID','Username'],axis=1,inplace=True)
test['Tag'] = le.fit_transform(test['Tag'])
test_pred = regr.predict(test)
ans = pd.DataFrame({'ID' : ID, 'Upvotes' : test_pred})
sub = ans.sort_values(by=['ID'])
print(sub)
file_name = 'new-rf__n-est_{}__max-depth_{}__r-s_{}.csv'.format(est,md,rs)
sub.to_csv(file_name, index=False)
