# Handwritten-Digit-Recognition
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_digits
dataset = load_digits()
X = dataset.data
y = dataset.target
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.3)

some_digit = X[270]
some_digit_image = some_digit.reshape(8,8)
plt.imshow(some_digit_image)
plt.show()
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X , y)
from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier(max_depth = 10)
dtf.fit(X , y)
dtf.score(X , y)
print(dtf.predict(X[[270],0:64]))
