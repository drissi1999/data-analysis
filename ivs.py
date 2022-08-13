#!/usr/bin/env python
# coding: utf-8

# In[141]:


#les bibliothèques Scikit-learn, Pandas et Matplotlib 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

df = pd.read_csv("C:/Users/driss/Downloads/games.csv")


# In[80]:


df.head()


# In[81]:


#1- Afficher les noms des attributs du dataset


# In[49]:


for col in df.columns:
    print(col)


# In[8]:


#2- Afficher l’histogramme de toutes les notes de la colonne average_rating


# In[50]:


df.average_rating.plot.hist()
plt.xlabel('average rating')
plt.title('l’histogramme de toutes les notes de la colonne average_rating')


# In[20]:


#3- Afficher la première ligne du dataset games avec des notes égales à 0


# In[51]:


df[ df["average_rating"]==0].head(1)


# In[36]:


#4- Afficher la première ligne de tous les jeux de sociétés dont la note est supérieure à 0


# In[52]:


df[ df["average_rating"] > 0].head(1)


# In[38]:


#5- Supprimer les jeux de société ne possédant aucune review


# In[127]:


df.dropna(subset = ['users_rated'])
df.isnull().sum()
df.dropna()


# In[56]:


#6- Utiliser K-means pour déterminer les différents clusters en initialisant le nombre de clusters à 5 (n_clusters = 5 et random_state=1)


# In[201]:


features = list(df.columns)[-10:]
_df = df[features]
clustering_kmeans = KMeans(n_clusters=5, random_state=1)
y_predict = clustering_kmeans.fit_predict(_df)
_df['clusters'] = y_predict


# In[196]:


# we got 5 clusters !
y_predict


# In[197]:


_df.head()


# In[198]:


clustering_kmeans.cluster_centers_


# In[ ]:


7-#Appliquer une analyse en composantes principales (Principle Components Analysis ou PCA) pour essayer de transformer nos données de jeux de société en deux dimensions, ou colonnes, afin de pouvoir facilement les tracer (total_owners et total_traders)


# In[204]:


pca_num_components = 2

reduced_data = PCA(n_components=pca_num_components).fit_transform(_df)
results = pd.DataFrame(reduced_data,columns=['total_owners','total_traders'])
results


# In[133]:


#8- Générer et afficher un graphique à nuage de points pour chaque type de jeux de société, à partir des clusters


# In[209]:


df1 = _df[_df['clusters'] == 0]
df2 = _df[_df['clusters'] == 1]
df3 = _df[_df['clusters'] == 2]
df4 = _df[_df['clusters'] == 3]
df5 = _df[_df['clusters'] == 4]

plt.scatter(df1.total_owners,df1.total_traders,color='green')
plt.scatter(df2.total_owners,df2.total_traders,color='red')
plt.scatter(df3.total_owners,df3.total_traders,color='blue')
plt.scatter(df4.total_owners,df4.total_traders,color='pink')
plt.scatter(df5.total_owners,df5.total_traders,color='yellow')

plt.xlabel('total_owners')
plt.ylabel('total_traders')
plt.show()


# In[210]:


#9- L’objectif est de pouvoir prédire average_rating et de ce fait il faut calculer, en premier temps, de façon combinatoire la corrélation entre la colone average_rating et les autres colonnes. Calculer et afficher les valeurs de corrélation


# In[213]:


df.corr()['average_rating']


# In[214]:


#10- Supprimer la colonne bayes_average_rating car elle dépend dans son calcul de average_rating ainsi que les colonnes non numériques comme type et name


# In[261]:


df.drop(['bayes_average_rating','name','type'], axis=1, inplace=True)
df.dropna(axis =0 ,inplace=True)


# In[263]:


df


# In[265]:


#11- Séparer le dataset en deux parties : données d’apprentissage 80 % données de test 20 %
from sklearn.model_selection import train_test_split

X_train, X_test ,Y_train, Y_test = train_test_split(df,df,train_size=0.8,test_size=0.2)


# In[255]:


print(X_train, X_test ,Y_train, Y_test)


# In[242]:


#12- Effectuer un apprentissage en utilisant la méthode de régression linéaire et ensuite calculer et afficher l’erreur de prédiction


# In[267]:


from sklearn.linear_model import LinearRegression

linerg = LinearRegression()


# In[268]:


linerg.fit(X_train,Y_train)


# In[269]:


y_predict = linerg.predict(X_test)


# In[270]:


y_predict


# In[274]:


import numpy as np
np.mean(Y_test - y_predict)**2 #mean des erreurs 


# In[275]:


#erreurs predct

from sklearn.metrics import  mean_squared_error

mean_squared_error(Y_test,y_predict)


# In[ ]:




