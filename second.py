import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

plt.close('all')
df = pd.read_csv(r'train_NIR5Yl1.csv')

print(df, df.describe())

y = df['Upvotes']
x1 = df[['Views','Answers']]

x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
reg = results.summary()
print(reg)

df_test = pd.read_csv(r'test_8i3B3FC.csv')

df_test['Upvotes'] = 0.0199*df_test['Views'] - 167.8726 - 21.4263*df_test['Answers']
print(df_test)

sort = df_test.sort_values(by=['ID'])
print(sort)

ans = sort[['ID','Upvotes']]
ans[ans<0]=0
print(ans)
ans.to_csv('second_sub.csv', index=False)
