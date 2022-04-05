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

# Drop rows that have a NaN in them
df = df.dropna()

# Remove features we don't want
# In this case, we remove non-float data as that will mess up the student's code
df.pop('TimeStamp')
df.pop('SourceIP')
df.pop('DestinationIP')

# Make a copy here just for visualization purposes
original_df = df.copy()

# Extract the labels from the dataframe and encode them to integers
df_labels = df.pop('Label')
label_encoder = LabelEncoder()
df_labels = label_encoder.fit_transform(df_labels)

# Prepare arrays to split into training and testing sets
x_features = df.values
y_labels = np.array(df_labels).T

# Split into training (70%) and testing (30%)
x_train, x_test, y_train, y_test = train_test_split(x_features, y_labels, train_size=0.7, random_state=1738, shuffle=True)

print(f"Training Set={x_train.shape}, Testing Set={x_test.shape}")

