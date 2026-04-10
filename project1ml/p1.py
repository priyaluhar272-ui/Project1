import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
housing= pd.read_csv('dataset.csv')
print(housing)
# # print(housing.head(5))
# # print(housing.info())
# # print(housing['4.CHAS'].value_counts())
# a=housing.columns = housing.columns.str.strip()
# print(a)
# print(housing['4. CHAS'].value_counts())
# print(housing.describe())

# housing.hist(bins=50,figsize=(20,15))
# plt.show()
# print(housing)
def split_train_test(data,test_ratio):
    shuffled=np.random.permutation(len(data))
    test_set_size=int(len(data) * test_ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]
train_set,test_set=split_train_test(housing,0.2)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}")