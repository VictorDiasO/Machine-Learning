import numpy as np
import pandas as pd

database = pd.read_csv('svbr.csv', delimiter=';')

X = database.iloc[:,:].values #get values of all rows(:) and all columns(:)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer = imputer.fit(X[:, 1:3])
X = imputer.transform(X[:,1:3]).astype(str)
X = np.insert(X, 0, database.iloc[:,0].values, axis=1)
print(X)