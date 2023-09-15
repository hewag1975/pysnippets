
from sklearn.datasets import fetch_openml

spam = fetch_openml('spambase')

spam.keys()
spam['data']
spam['target']
spam['feature_names']
spam['target_names']


## Train-test-split
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(
    spam['data'],
    spam['target'], 
    test_size=0.3
)


## Logistic regression
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(xtrain, y=ytrain)
lm.coef_

ptest = lm.predict(xtest)



