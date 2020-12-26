import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

plt.close('all')
data = pd.read_csv(r'train_NIR5Yl1.csv')

df = data.copy()
df['Tag'] = df['Tag'].map({'c':0,'j':1,'p':2,'i':3,'a':4,'s':5,'h':6,'o':7,'r':8,'x':9})
print(df, df.describe())
y = df['Upvotes']
x = df[['Views','Answers','Tag','Reputation']]
# ~ x_matrix = x.values.reshape(-1,1)
# ~ print(x_matrix.shape)
reg = LinearRegression()

# ~ reg.fit(x_matrix,y)
# ~ R_sq = reg.score(x_matrix,y)
reg.fit(x,y)
R_sq = reg.score(x,y)         #R^2
# ~ print(R_sq)

coef = reg.coef_
# ~ print(coef)

inter = reg.intercept_
# ~ print(inter)

df_test = pd.read_csv(r'test_8i3B3FC.csv')
df_test['Tag'] = df_test['Tag'].map({'c':0,'j':1,'p':2,'i':3,'a':4,'s':5,'h':6,'o':7,'r':8,'x':9})
x_test = df_test[['Views','Answers','Tag','Reputation']]
# ~ x_matrix_new = x_new.values.reshape(-1,1)
# ~ print(x_matrix_new)
# ~ ans = reg.predict(x_matrix_new)
ans = reg.predict(x_test)

# ~ print(ans.shape, ans)

sort = df_test.sort_values(by=['ID'])
idx = sort[['ID']]
final_ans = pd.DataFrame(data=idx, columns = ['ID'])

ans[ans<0]=0
final_ans['Upvotes'] = ans
print(final_ans)
# ~ final_ans.to_csv('sk_all.csv', index=False)

from sklearn.feature_selection import f_regression
f_regression(x,y)
p_values = f_regression(x,y)[1]
p_values.round(3)

reg_summary = pd.DataFrame(data = x.columns.values,columns =['Features'])
reg_summary['Coefficients'] = reg.coef_
reg_summary['p-values'] = p_values.round(3)
print(reg_summary)
