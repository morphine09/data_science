#!/usr/bin/env python
# coding: utf-8

# In[10]:


pip install utils_py


# In[12]:


import sys
sys.path.append("/path/to/directory/containing/utils")


# In[14]:


import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


df = pd.read_csv("https://raw.githubusercontent.com/siddiquiamir/Transfer-Learning-TensorFlow/main/house_data.csv")
df


# In[16]:


df.head()


# In[18]:


df.shape


# In[20]:


df.hist("price")
plt.show()


# In[21]:


df.isna()


# In[22]:


df.isna().sum()


# In[23]:


df = df.iloc[:,1:]
df_norm = (df - df.mean()) / df.std()
df_norm.head()


# In[25]:


X = df_norm.iloc[:, :5]
X.head()


# In[26]:


Y = df_norm.iloc[:, -1]
Y.head()


# In[27]:


X_arr = X.values
Y_arr = Y.values


# In[28]:


X_arr


# In[29]:


Y_arr


# # TRAIN TEST SPLIT
# 

# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X_arr, Y_arr, test_size = 0.01, shuffle = True, random_state=1)

print('X_train shape: ', X_train.shape)
print('y_train shape: ', y_train.shape)
print('X_test shape: ', X_test.shape)
print('y_test shape: ', y_test.shape)


# In[31]:


def get_model():
    
    model = Sequential([
        Dense(10, input_shape = (5,), activation = 'relu'),
        Dense(20, activation = 'relu'),
        Dense(5, activation = 'relu'),
        Dense(1)
    ])

    model.compile(
        loss='mse',
        optimizer='adadelta'
    )
    
    return model


# In[32]:


model = get_model()
model.summary()


# 
# # Model Training

# In[33]:


model = get_model()

# this prediction is before training the model
preds_on_untrained = model.predict(X_test)


# In[34]:


# Train model and store in the object history
history = model.fit(
    X_train, y_train,
    validation_data = (X_test, y_test),
    epochs = 1000
)


# # Prediction

# In[37]:


# make predictions on the trained model
preds_on_trained = model.predict(X_test)


# In[42]:


import matplotlib.pyplot as plt
import numpy as np

def compare_predictions(preds_on_untrained, preds_on_trained, y_test):
    """
    Compare and visualize predictions on untrained and trained models.

    Parameters:
    - preds_on_untrained: Predictions made by an untrained model.
    - preds_on_trained: Predictions made by a trained model.
    - y_test: True labels from the test set.
    """
    plt.figure(figsize=(12, 6))

    # Plot predictions on untrained model
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, preds_on_untrained, color='blue', alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')
    plt.title('Predictions on Untrained Model')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')

    # Plot predictions on trained model
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, preds_on_trained, color='green', alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')
    plt.title('Predictions on Trained Model')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')

    plt.tight_layout()
    plt.show()

# Example usage:
# compare_predictions(preds_on_untrained, preds_on


# In[43]:


compare_predictions(preds_on_untrained, preds_on_trained, y_test)


# In[ ]:





# In[40]:


import matplotlib.pyplot as plt

def plot_loss(history):
    """
    Plots the training and validation loss over epochs.

    Parameters:
    - history: History object returned by the fit method of a Keras model.
    """
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Example usage:
# Assuming you have already trained your model and obtained the history object
# history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
# plot_loss(history)


# In[41]:


plot_loss(history)

