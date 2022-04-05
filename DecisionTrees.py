import os
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("doh_dataset.zip", compression='gzip')

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train = np.float32(x_train/255).reshape(x_train.shape[0],-1)
x_test = np.float32(x_test/255).reshape(x_test.shape[0],-1)

#1: entropy
model = DecisionTreeClassifier(criterion='entropy')
model = model.fit(x_train, y_train)
pred = model.predict(x_test)
accuracy = accuracy_score(y_test, pred)
print(f'Accuracy {accuracy:.4}')
cm = confusion_matrix(y_test, pred)
print(cm)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

fig, ax = plt.subplots(figsize=(8,8))
tree.plot_tree(model, fontsize=10, ax=ax,class_names=class_names)
fig.suptitle("Decision tree", fontsize=14)

#use gini instead of entropy.
model = DecisionTreeClassifier(criterion='gini')
model = model.fit(x_train, y_train)
pred = model.predict(x_test)
accuracy = accuracy_score(y_test, pred)
print(f'Accuracy {accuracy:.4}')
cm = confusion_matrix(y_test, pred)
print(cm)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

fig, ax = plt.subplots(figsize=(8,8))
tree.plot_tree(model, fontsize=10, ax=ax,class_names=class_names)
fig.suptitle("Decision tree", fontsize=14)

#use gini instead of entropy.
model = DecisionTreeClassifier(criterion='gini')
model = model.fit(x_train, y_train)
pred = model.predict(x_test)
accuracy = accuracy_score(y_test, pred)
print(f'Accuracy {accuracy:.4}')
cm = confusion_matrix(y_test, pred)
print(cm)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

fig, ax = plt.subplots(figsize=(8,8))
tree.plot_tree(model, fontsize=10, ax=ax,class_names=class_names)
fig.suptitle("Decision tree", fontsize=14)

#3: Find the best performing decision tree (use accuracy - evaluated on the test dataset) 
#of depth of 20 or less, display it, and print the confusion matrix and accuracy 
#on the test dataset. Use entropy.

best_model = DecisionTreeClassifier(criterion='entropy')
best_accuracy = 0

max_depths = list(np.arange(15,21))+[None]
for max_depth in max_depths:
  model = DecisionTreeClassifier(criterion='entropy', max_depth = max_depth)
  model = model.fit(x_train, y_train)
  pred = model.predict(x_test)
  accuracy = accuracy_score(y_test, pred)

  if (best_accuracy < accuracy):
    best_model = model
    best_accuracy = accuracy
    best_max_depth = max_depth

  print('Max_depth = {};, accuracy = {:7.4f}'.format(max_depth,accuracy))

print(f'Accuracy {best_accuracy:.4}')
print("best_max_depth : ", best_max_depth)
cm = confusion_matrix(y_test, pred)
print(cm)