import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import load_iris
iris = load_iris()
print(dir(iris))

data = pd.DataFrame(iris.data, columns = iris.feature_names)
print(data.head())
data['target'] = iris.target

data['label'] = data.target.apply(lambda x: iris.target_names[x])
print(data.tail())

#make variables: features and target

X = data.iloc[:, 0:4]
y = data.iloc[:, 4]
print(y.head())



from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model = MultinomialNB()
model.fit(X,y)
print(model.score(x_test, y_test))

print(model.predict([[3,5,5,2]]))

print(data.label.value_counts())