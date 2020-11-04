import pandas as pd

df = pd.read_csv(r'test_8i3B3FC.csv')
print (df)

df1 = pd.read_csv(r'mean_max_min_upvotes_Tag.csv', index_col=[0])  ## To generate this file code is written below

print (df1)
sort_ID = df.ID.sort_values()
print(sort_ID)

d = pd.DataFrame()

for i in sort_ID:
	x = df.loc[df['ID'] == i, 'Tag']   #x is tag
	y = df1.loc[df1['Tag'].isin(x), 'Mean upvotes']       #y is mean-upvote
	# ~ print(f'\n{y}')
	temp = pd.DataFrame.from_records([{'ID':i, 'Upvotes':float(y)}])
	d = (pd.concat([d, temp])).reset_index(drop = True)
	# ~ print(temp)

print(d)
d.to_csv('sub_Tag.csv', index=False)

##################
#df = pd.read_csv(r'C:\Users\Khushbu\Downloads\train_NIR5Yl1.csv')

#freq = df.Tag.value_counts()

#d = pd.DataFrame()

f#or i in freq.index:
#temp = pd.DataFrame.from_records([{'Tag':i,
#	'No. of ques':freq[i],
#	'Mean upvotes':df.loc[df['Tag'] == i, 'Upvotes'].mean()
	#}])
	#d = (pd.concat([d, temp])).reset_index(drop = True)
#print(d)
#d.to_csv('mean_max_min_upvotes_Tag.csv')

##############
