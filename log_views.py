import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set()

raw_data = pd.read_csv('train_NIR5Yl1.csv')
print(raw_data.head())
print(raw_data.describe(include='all'))

r = raw_data['Views'].quantile(0.99)
data_1 = raw_data[raw_data['Views']<r]
print(data_1.describe(include='all'))

data_cleaned = data_1.reset_index(drop=True)

log_Views = np.log(data_cleaned['Views'])
data_cleaned['log_Views'] = log_Views

data_with_dummies = pd.get_dummies(data_cleaned, drop_first=True)
data_preprocessed = data_with_dummies
targets = data_preprocessed['Upvotes']
inputs = data_preprocessed.drop(['Upvotes','Views','ID','Username'],axis=1)

reg = LinearRegression()
reg.fit(inputs, targets)
y_hat = reg.predict(inputs)
plt.scatter(targets, y_hat)
plt.xlabel('Targets (y_train)',size=18)
plt.ylabel('Predictions (y_hat)',size=18)

reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_
print(reg_summary)
print(reg.intercept_)

df_test = pd.read_csv(r'test_8i3B3FC.csv')
sort = df_test.sort_values(by=['ID'])
test_data_with_dummies = pd.get_dummies(sort, drop_first=True)

log_Views = np.log(test_data_with_dummies['Views'])
test_data_with_dummies['log_Views'] = log_Views
inputs_test = test_data_with_dummies.drop(['ID','Username','Views'],axis=1)
ans = reg.predict(inputs_test)
print(ans)

idx = sort[['ID']]
final_ans = pd.DataFrame(data=idx, columns = ['ID'])
ans[ans<0]=0
final_ans['Upvotes'] = ans
print(final_ans)
final_ans.to_csv('log_Views.csv', index=False)
plt.show()





