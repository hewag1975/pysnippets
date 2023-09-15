import pandas as pd
import numpy as np
from sklearn import datasets

## toy datasets
## https://scikit-learn.org/stable/datasets/toy_dataset.html

## iris
iris = datasets.load_iris()
iris.keys()

iris['data']
iris['target']
iris['feature_names']


### convert numpy to pandas
iris_df = pd.DataFrame(iris['data'])
iris_df.columns = iris['feature_names']

nms = iris_df.columns
nms = nms.str.replace(" (cm)", repl="")
nms = nms.str.replace(" ", repl="_")
iris_df.columns = nms

iris_df.iloc[0:3]
iris_df.iloc[:3]
iris_df.iloc[3:]

## digits
digits = datasets.load_digits()
digits['data'].shape
digits['target'].shape

from matplotlib import pyplot

pix = digits['data']
# pix = np.reshape(pix[0], newshape=(8, 8))
pix = np.reshape(pix, newshape=(1797, 8, 8))

pix.shape

pyplot.plot()
pyplot.imshow(pix[0], cmap=pyplot.get_cmap('gray'))
pyplot.show()


