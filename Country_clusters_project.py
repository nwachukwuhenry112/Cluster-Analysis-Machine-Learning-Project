#!/usr/bin/env python
# coding: utf-8

# In[1]:


# the data used for this study was collected from Udemy data science exercise..
# Henry Nwachukwu Project 1 Date submitted: 20:10:2020
# Project Summary: TO group countries based on their Longitude and Latitude


# In[2]:


pwd


# In[3]:


# Import Relevant libraries for the studies:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# In[4]:


country_data=pd.read_csv('country.csv')


# In[5]:


country_data.head(5)


# In[6]:


country_data.head(5)


# In[7]:


# The Approach that will be utilize in this study is KMeans
# KMeans is sensitive to outliers, Standardize data and missing value 
# and.. So effort will be made during EDA to ensure the data is good for analysis.


# In[8]:


# checking for missing Values
country_data.isnull().sum()


# In[9]:


# checking for outliers
country_data.describe(include = 'all')


# In[10]:


# Defining boundries for removing Outliers using the IQR ( Inter-quartile-Range)Approach.


# In[11]:


# Longitude 
Q3=47.58 
Q1=-36.43
IQR=Q3-Q1
lower_fence= Q1-(1.5*IQR)
Upper_fence=Q3+(1.5*IQR)


# In[12]:


lower_fence


# In[13]:


Upper_fence


# In[14]:


countries_above = country_data[ (country_data.Longitude > 173)]


# In[15]:


countries_above


# In[16]:


countries_below = country_data[ (country_data.Longitude < -162.445)]


# In[17]:


countries_below


# In[18]:


# the above countries fall off the acceptable range and will for sure have different clusters ( using the Longitude values)
# But for the sake of this study all will be Clusters.


# In[19]:


# To view the box plot
Longitute_array = country_data["Longitude"].to_numpy()


# In[20]:


# Creating Plot
plt.boxplot(Longitute_array,patch_artist=True,labels=['Longitude'])
#Show plot
plt.show()


# In[21]:


# Comparing box boxp plot for Longitude and Latitude
array =  country_data[["Longitude","Latitude"]].to_numpy()


# In[22]:


#Comparing both Longitude and Latitude box plot for outliers
# Creating Plot
plt.boxplot(array,patch_artist=True,labels=['Longitude','Latitude'])
#Show plot
plt.show()


# In[23]:


X_unscaled = country_data.copy()


# In[24]:


X_unscaled


# In[25]:


X_unscaled_study = X_unscaled.iloc[:,1:3]


# In[26]:


# Now am going to scale X_unscaled_study using MinMaxScaler


# In[27]:


scaler = MinMaxScaler()
scaler.fit(X_unscaled_study)


# In[28]:


# this approach shows the data in array
X_scaled_study = scaler.fit_transform(X_unscaled_study)


# In[29]:


# I will be using the X_scaled_study_aba for further studies
X_scaled_study_aba = pd.DataFrame(data=scaler.transform(X_unscaled_study ),columns=['Longitude','Latitude']) 


# In[30]:


X_scaled_study_aba 


# In[31]:


# I assume a cluster size of 4 at this early stage, after it will be updated with the WCSS
# Clustering
kmeans_4 = KMeans(4)


# In[32]:


kmeans_4.fit(X_scaled_study_aba)


# In[33]:


# Clustering Result
identified_clusters = kmeans_4.fit_predict(X_scaled_study_aba)
identified_clusters


# In[34]:


# Adding the cluster column to the data set and visualising it
data_with_clusters = country_data.copy()
data_with_clusters['Clustering_4'] = identified_clusters
data_with_clusters


# In[35]:


data_with_clusters.tail()


# In[36]:


# 2 Key ways of viewing Clusters Analysis plots: Scatter plot & Heat map (Dendogram)
# Scatter plot uses the unscaled axis vs the clusters generated from the scaled parameters.

#Scatter Plot grouping names with the intial assume cluster size of 4.( will be updated after optimal size has being estimayted)
plt.scatter(country_data['Longitude'], country_data['Latitude'],c=data_with_clusters['Clustering_4'], cmap = 'rainbow')
plt.xlim(-180,180)
plt.ylim(-90, 90)
plt.show()

# Not satisfied with this cluster so, How do i find the optimal number of clusters?


# In[37]:


# How do i find the optimal number of clusters?
# Write a loop that calculates and saves the WCSS for any number of clusters from 1 up to 20
# Determing the optimal number of clusters using the Elbow Method
# optimal number is the minimal WCSS from plot of within_cluster sum of squares vs number of clusters.


# In[38]:


# get the wscc for the current solution
kmeans_4.inertia_
wscc = []
for i in range(1,21):
    kmeans_4 = KMeans(i)
    kmeans_4.fit(X_scaled_study_aba)
    wscc_iter = kmeans_4.inertia_
    wscc.append(wscc_iter)
    


# In[39]:


wscc


# In[40]:


# Plot the Elbow to etimate miminal wscc that will be use as the optimal number os clusters.
number_clusters = range(1,21)
plt.plot(number_clusters,wscc)
plt.title('Elbow Method')
plt.xlabel('number_clusters')
plt.ylabel('within_clusters sum of squares')

# Based on the Elbow Curve, I am going to ( 5 , 8, 10), and see one that gives me more deseireable and interpretable plot.


# In[41]:


# Assuming 5 as the desireable minimum wscc
kmeans_5 = KMeans(5)


# In[42]:


kmeans_5.fit(X_scaled_study_aba)


# In[43]:


# Clustering Result
identified_clusters_5 = kmeans_5.fit_predict(X_scaled_study_aba)
identified_clusters_5


# In[44]:


# Adding the cluster column to the data set and visualising it
data_with_clusters = country_data.copy()
data_with_clusters['Clustering_5'] = identified_clusters_5
data_with_clusters


# In[45]:


#Scatter Plot grouping names with 5 clusters
plt.scatter(country_data['Longitude'], country_data['Latitude'],c=data_with_clusters['Clustering_5'], cmap = 'rainbow')
plt.xlim(-180,180)
plt.ylim(-90, 90)
plt.show()


# In[46]:


# Assuming 8 as the desireable minimum wscc
kmeans_8 = KMeans(8)


# In[47]:


kmeans_8.fit(X_scaled_study_aba)


# In[48]:


# Clustering Result
identified_clusters_8 = kmeans_8.fit_predict(X_scaled_study_aba)
identified_clusters_8


# In[49]:


# Adding the cluster column to the data set and visualising it
data_with_clusters = country_data.copy()
data_with_clusters['Clustering_8'] = identified_clusters_8
data_with_clusters


# In[50]:


#Scatter Plot grouping names with 8 clusters
plt.scatter(country_data['Longitude'], country_data['Latitude'],c=data_with_clusters['Clustering_8'], cmap = 'rainbow')
plt.xlim(-180,180)
plt.ylim(-90, 90)
plt.show()


# In[51]:


# Assuming 10 as the desireable minimum wscc
kmeans_10 = KMeans(10)


# In[52]:


kmeans_10.fit(X_scaled_study_aba)


# In[53]:


# Clustering Result
identified_clusters_10 = kmeans_10.fit_predict(X_scaled_study_aba)
identified_clusters_10


# In[54]:


# Adding the cluster column to the data set and visualising it
data_with_clusters = country_data.copy()
data_with_clusters['Clustering_10'] = identified_clusters_10
data_with_clusters


# In[55]:


#Scatter Plot grouping names with 10 clusters
plt.scatter(country_data['Longitude'], country_data['Latitude'],c=data_with_clusters['Clustering_10'], cmap = 'rainbow')
plt.xlim(-180,180)
plt.ylim(-90, 90)
plt.show()


# In[56]:


# from the comparism of the clusters plot. I strongly believe that the countries can best be grouped into 4 classes using their various longitude and latitude.


# In[57]:


# for better visualization of the grouping done for these countries am going to use Heatmap( Dendogram)
# this plot uses the scaled values of longitude and latitude and the countries they grouped.


# In[58]:


countries=country_data.iloc[:,0]
countries


# In[59]:


X_scaled_study_aba


# In[60]:


Countries_grouped= pd.concat([countries, X_scaled_study_aba], axis=1)


# In[61]:


Countries_grouped


# In[62]:


Country_Grouped = Countries_grouped.set_index('name')
Country_Grouped


# In[63]:


# Using seaborn to plot the Heatmap(Dendogram)
Henry = sns.clustermap(Country_Grouped,cmap = "vlag")


# In[64]:


# Now am going to group these countries into various groups( A,B,C & D)
Summarized_clusters= pd.DataFrame(identified_clusters,columns=['names_groups'])


# In[65]:


Summarized_clusters


# In[66]:


Summarized_clusters_GROUPS= pd.concat([Countries_grouped, Summarized_clusters], axis=1)


# In[67]:


GROUP_A = Summarized_clusters_GROUPS[ (Summarized_clusters_GROUPS.names_groups == 0)]


# In[68]:


GROUP_A


# In[69]:


GROUP_B = Summarized_clusters_GROUPS[ (Summarized_clusters_GROUPS.names_groups == 1)]


# In[70]:


GROUP_B


# In[71]:


GROUP_C = Summarized_clusters_GROUPS[ (Summarized_clusters_GROUPS.names_groups == 2)]


# In[72]:


GROUP_C 


# In[73]:


GROUP_D = Summarized_clusters_GROUPS[ (Summarized_clusters_GROUPS.names_groups == 3)]


# In[74]:


GROUP_D


# In[75]:


# Further effort will be made to join the Groups by rows and plot the heatmap for better clearer visualization.
Combined_groups= pd.concat([GROUP_A,GROUP_B,GROUP_C,GROUP_D], axis=0)


# In[76]:


Combined_groups


# In[77]:


Henry_Combined_groups=Combined_groups.copy()


# In[78]:


Henry_Combined_groups


# In[79]:


Henry_Combined_groups=Henry_Combined_groups.drop(columns=['names_groups'], axis=1)


# In[80]:


Henry_Combined_groups


# In[81]:


Henry_Combined_groups= Henry_Combined_groups.set_index('name')


# In[82]:


Henry_Combined_groups


# In[83]:


# Using seaborn to plot the Heatmap(Dendogram)
Henry_2 = sns.clustermap(Henry_Combined_groups,cmap = "vlag")


# In[84]:


# It will be nice to save the model so it can be use for Deployment.
# if any of the (5,8,10) had worked better, I would have gone back to refit the model using (kmean_x fit)
import pickle
with open ('model','wb')as file:
    pickle.dump(kmeans_4,file)


# In[85]:


# save the scaled model ie (the transformoperation).
with open ('scaler','wb')as file:
    pickle.dump(X_scaled_study_aba,file)


# In[86]:


# Saving the project using the module: .py file( file:download as python(.py)
# To deploy using the Custom-made module, all must be saved in the same folder including the new data sets.

# Thanks( Henry Nwachukwu: Machine Learning Analyst @ Evolveu Calgary IT Center.)

