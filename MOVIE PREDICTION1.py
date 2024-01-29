#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


import pandas as pd

movies = pd.read_csv(r"D:\Downloads\top10K-TMDB-movies.csv.zip")
movies


# In[4]:


movies.head(10)
#getting the details of the first ten movies


# In[5]:


movies.describe()
#to get basic information of the dataset which doesnt really help in data cleaning


# In[6]:


movies.info()
#getting in depth details of the dataset which becomes useful in data cleaning


# In[7]:


movies.isnull().sum()
#finding out null values to reject them later


# In[8]:


movies = movies[['id', 'title', 'overview', 'genre']]
movies
#keeping these four columns in the dataset because these four are the essential requirements for the predictive model


# In[9]:


movies['tags'] = movies['overview']+movies['genre']
movies
s


# In[10]:


new_data = movies.drop(columns = ['overview','genre'])
new_data
#dropping overview and genre because we have already created the combination of both in a new column named tags


# In[11]:


from sklearn.feature_extraction.text import CountVectorizer


# In[12]:


cv = CountVectorizer(max_features = 10000, stop_words = 'english')
cv
#max feautres is the number of movies we have in the dataset and stop words is the medium of language we want to keep in this training model


# In[13]:


vector=cv.fit_transform(new_data['tags'].values.astype('U')).toarray()
#transforming all the information into arrays and vector


# In[14]:


vector.shape


# In[15]:


from sklearn.metrics.pairwise import cosine_similarity
#cosine similarity finds the similarity between two sentences and groups them as one


# In[18]:


from sklearn.feature_extraction.text import CountVectorizer

max_features = 1000  # Adjust the number of features as needed
cv = CountVectorizer(max_features=max_features)
vector = cv.fit_transform(new_data['tags'].values.astype('U')).toarray()
#since the dataset was too big to vectorise, i had to use count vectoriser to fix the max features uptill 1000


# In[19]:


vector.shape


# In[20]:


from sklearn.metrics.pairwise import cosine_similarity


# In[21]:


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

chunk_size = 1000
num_samples = vector.shape[0]
similarity_matrix_path = "similarity_matrix.npy"

for i in range(0, num_samples, chunk_size):
    end_idx = min(i + chunk_size, num_samples)
    vector_chunk = vector[i:end_idx, :]
    
    if i == 0:
        similarity_matrix = cosine_similarity(vector_chunk, vector)
    else:
        similarity_chunk = cosine_similarity(vector_chunk, vector)
        similarity_matrix = np.maximum(similarity_matrix, similarity_chunk)

np.save(similarity_matrix_path, similarity_matrix)
#this might have hampered with the data readings since memory allocation was the biggest challege faced while using cosine similarity
#I divided the data into chunks in order to streamline the process but it might have caused alterations with the output of the prediction model




# In[22]:


similarity = cosine_similarity(vector)
similarity


# In[23]:


new_data[new_data['title']=="The Godfather"].index[0]


# In[24]:


distance = sorted(list(enumerate(similarity[2])), reverse = True, key = lambda vector:vector[1])
for i in distance [0:5]:
    print(new_data.iloc [i[0]].title)
    


# In[26]:


def recommend(movies):
    index = new_data[new_data['title'] == movies].index[0]
    distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector: vector[1])
    for i in distance[0:5]:
        print(new_data.iloc[i[0]].title)



# In[27]:


recommend("Batman Begins")


# In[30]:


recommend("The Shawshank Redemption")


# In[ ]:




