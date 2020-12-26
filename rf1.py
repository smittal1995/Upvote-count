from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

import numpy as np
import pandas as pd
import statsmodels.api as sm
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
# ~ X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
# ~ print(X,y)
est = 700
md = 13
rs = 2
regr = RandomForestRegressor(n_estimators=est ,max_depth=md, random_state=rs, verbose=2)
regr.fit(inputs_scaled, targets)
predictions = regr.predict(inputs_scaled)
errors = abs(predictions - targets)
# ~ print(errors)
# Calculate mean absolute percentage error (MAPE)
# ~ mape = 100 * (errors / targets)
# Calculate and display accuracy
# ~ accuracy = 100 - np.mean(mape)
# ~ print('Accuracy:', round(accuracy, 2), '%.')

df_test = pd.read_csv(r'test_8i3B3FC.csv')
sort = df_test.sort_values(by=['ID'])
test_data_with_dummies = pd.get_dummies(sort, drop_first=True)
input_test = test_data_with_dummies.drop(['ID','Username'],axis=1)
scaler.fit(input_test)
input_scaled_test = scaler.transform(input_test)
ans = regr.predict(input_scaled_test)
idx = sort[['ID']]
final_ans = pd.DataFrame(data=idx, columns = ['ID'])
ans[ans<0]=0
final_ans['Upvotes'] = ans

print(final_ans)
file_name = 'n-est_{}__max-depth_{}__r-s_{}.csv'.format(est,md,rs)
final_ans.to_csv(file_name, index=False)




