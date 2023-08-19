import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import os

data_path = os.path.join(os.getcwd(), "BikeDataset/day.csv")
data_day = pd.read_csv(data_path)
st.title('Bike Sharing Data Analysis Dashboard')
analysis_option = st.sidebar.selectbox('Choose Analysis', ['Trends Over Time', 'Weather Correlation', 'Clustering'])
st.set_option('deprecation.showPyplotGlobalUse', False)

# Main content based on user selection
if analysis_option == 'Trends Over Time':
    st.header('Trends Over Time Analysis')
    
    # Visualization for Daily Counts
    st.subheader('Daily Bike Rental Counts')
    daily_counts = data_day.groupby('dteday')['cnt'].sum()
    st.line_chart(daily_counts)
    
    # Visualization for Yearly Counts
    st.subheader('Yearly Bike Rental Counts')
    yearly_counts = data_day.groupby('yr')['cnt'].sum()
    yearly_counts.index = ['2011', '2012']
    st.bar_chart(yearly_counts)
    
elif analysis_option == 'Weather Correlation':
    st.header('Weather Correlation Analysis')
    
    # Visualization for Weather Correlation
    st.subheader('Average Rental Counts by Weather Condition')
    weather_counts = data_day.groupby('weathersit')['cnt'].mean()
    st.bar_chart(weather_counts)

elif analysis_option == 'Clustering':
    st.header('K-Means Clustering Analysis')
    cluster_data = data_day[['temp', 'hum']]
    st.sidebar.subheader("K-Means Clustering")
    num_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=3)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_data['cluster'] = kmeans.fit_predict(cluster_data)
    st.subheader("Clustered Scatter Plot")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=cluster_data, x='temp', y='hum', hue='cluster', palette='viridis')
    plt.xlabel('Temperature')
    plt.ylabel('Humidity')
    st.pyplot()
    st.subheader("Cluster Centers")
    st.dataframe(pd.DataFrame(kmeans.cluster_centers_, columns=['temp', 'hum']))
    st.subheader("Cluster Counts")
    cluster_counts = cluster_data['cluster'].value_counts()
    st.bar_chart(cluster_counts)

st.sidebar.markdown("Created by Elvaret")