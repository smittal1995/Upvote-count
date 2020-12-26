# Necessary imports 
import numpy as np 
import pandas as pd 
import xgboost as xg 

from sklearn.metrics import mean_squared_error as MSE 

# Load the data 
raw_data = pd.read_csv('train_NIR5Yl1.csv')

data_preprocessed = pd.get_dummies(raw_data, drop_first=True)

targets = data_preprocessed['Upvotes']
inputs = data_preprocessed.drop(['Upvotes','ID','Username'],axis=1) 

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)

# Instantiation 
xgb_r = xg.XGBRegressor(objective ='reg:linear', 
				n_estimators = 10, seed = 123) 

# Fitting the model 
xgb_r.fit(inputs_scaled, targets) 

pred = xgb_r.predict(input_test) 

# RMSE Computation 
rmse = np.sqrt(MSE(targets, pred)) 
print("RMSE : % f" %(rmse)) 

df_test = pd.read_csv(r'test_8i3B3FC.csv')
sort = df_test.sort_values(by=['ID'])
test_data_with_dummies = pd.get_dummies(sort, drop_first=True)
input_test = test_data_with_dummies.drop(['ID','Username'],axis=1)
scaler.fit(input_test)
input_scaled_test = scaler.transform(input_test)

# Predict the model 
ans = xgb_r.predict(input_scaled_test) 
idx = sort[['ID']]
final_ans = pd.DataFrame(data=idx, columns = ['ID'])
ans[ans<0]=0
final_ans['Upvotes'] = ans
# RMSE Computation 
# ~ rmse = np.sqrt(MSE(test_y, pred)) 
# ~ print("RMSE : % f" %(rmse)) 
final_ans.to_csv('xg1.csv', index=False)
