import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

penguins = pd.read_csv('penguins_cleaned.csv')

df = penguins.copy()

tagret = 'species'
encode = ['sex', 'island']

# using one hot encoding to represent the categorical data
for col in encode:
    dummy = pd.get_dummies(df[col],prefix=col) # for every categorial col in datafram get their dummy representation
    df = pd.concat([df,dummy],axis=1)
    del df[col] #delete the real column

# dictionary to represent every unique element in target column
target_mapper = {
    'Adelie': 0,
    'Chinstrap':1,
    'Gentoo':2
}

def target_encode(val):
    return target_mapper[val] # this function returns the corresponding values of the keys

df['species'] = df['species'].apply(target_encode)

X = df.drop('species',axis = 1)
y = df['species']

clf = RandomForestClassifier()
clf.fit(X,y)

pickle.dump(clf,open('penguins_clf.pkl','wb'))
