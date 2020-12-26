import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set()

raw_data = pd.read_csv('train_NIR5Yl1.csv')

data_preprocessed = pd.get_dummies(raw_data, drop_first=True)

targets = data_preprocessed['Upvotes']
inputs = data_preprocessed.drop(['Upvotes','ID','Username'],axis=1)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)
reg = LinearRegression()
reg.fit(inputs_scaled, targets)
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
input_test = test_data_with_dummies.drop(['ID','Username'],axis=1)
scaler.fit(input_test)
inputs_scaled_test = scaler.transform(input_test)

ans = reg.predict(inputs_scaled_test)
idx = sort[['ID']]
final_ans = pd.DataFrame(data=idx, columns = ['ID'])
ans[ans<0]=0
final_ans['Upvotes'] = ans

print(final_ans)
final_ans.to_csv('scaled.csv', index=False)

plt.show()





