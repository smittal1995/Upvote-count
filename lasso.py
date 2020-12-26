import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, Binarizer
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score

train = pd.read_csv('train_NIR5Yl1.csv')
train.head()

train.drop(['ID','Username'],axis=1,inplace=True)

bn = Binarizer(threshold=5)
pd_watched = bn.transform([train['Answers']])[0]
train['pd_watched'] = pd_watched

le = LabelEncoder()
train['Tag'] = le.fit_transform(train['Tag'])
print(train.head())


X=train.drop('Upvotes',axis=1)
y=train['Upvotes']

std=StandardScaler()
X_scaled=pd.DataFrame(std.fit_transform(X),columns=X.columns,index=X.index)

ts = 0.24
rs = 205
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=ts, random_state=rs)

print(X_train.head())
print(X_val.head())

poly_reg=PolynomialFeatures(degree=4,include_bias=True,interaction_only=False)
X_poly_train = poly_reg.fit_transform(X_train)
X_poly_train = pd.DataFrame(X_poly_train)
X_poly_val = poly_reg.fit_transform(X_val)
X_poly_val = pd.DataFrame(X_poly_val)

alp = 0.027
lin_reg_1 = linear_model.LassoLars(alpha=alp,max_iter=150)
lin_reg_1.fit(X_poly_train,y_train)

pred_train = lin_reg_1.predict(X_poly_train)
print('Train R2:',r2_score(y_train, pred_train))
print('Train RMSE:',np.sqrt(mean_squared_error(y_train, pred_train)))

pred_val = lin_reg_1.predict(X_poly_val)
print('Val R2:',r2_score(y_val, pred_val))
print('Val RMSE:',np.sqrt(mean_squared_error(y_val, pred_val)))

test = pd.read_csv('test_8i3B3FC.csv')
ID = test['ID']
test.drop(['ID','Username'],axis=1,inplace=True)
test['Tag'] = le.fit_transform(test['Tag'])
pd_watched = bn.transform([test['Answers']])[0]
test['pd_watched'] = pd_watched

test_scaled=pd.DataFrame(std.fit_transform(test),columns=test.columns,index=test.index)
test_poly = poly_reg.fit_transform(test_scaled)
test_poly = pd.DataFrame(test_poly)
test_pred = lin_reg_1.predict(test_poly)
test_pred = abs(test_pred)

ans = pd.DataFrame({'ID' : ID, 'Upvotes' : test_pred})
sub = ans.sort_values(by=['ID'])
print(sub)
file_name = '5-lasso__ts_{}__rs_{}__alpha_{}.csv'.format(ts,rs,alp)
sub.to_csv(file_name, index=False)
