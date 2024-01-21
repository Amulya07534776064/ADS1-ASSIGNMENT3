import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import t
from scipy.optimize import curve_fit
from sklearn.metrics import silhouette_score

#Read weather data from a CSV file.
def read_weather_data(file_path, nrows=None, usecols=None):
    """
    Read weather data from a CSV file.
    """
    return pd.read_csv(file_path, nrows=nrows, usecols=usecols)


def err_ranges(x, y, fit, confidence=0.95):
    """
    Calculate the error ranges for the polynomial fit.
    """
    y_pred = np.polyval(fit, x)
    residuals = y - y_pred
    ss_res = np.sum(residuals**2)
    n = len(y)
    p = len(fit)
    std_error = np.sqrt(ss_res / (n - p)) * np.sqrt(1 + 1/n + ((x - np.mean(x))**2 / np.sum((x - np.mean(x))**2)))
    t_val = t.ppf((1 + confidence) / 2, n - p)
    ci = t_val * std_error
    return y_pred - ci, y_pred + ci

def calculate_wcss(data, max_k):
    """
    Calculate the sum of squared distances for different clusters.
    """
    wcss = []
    for n in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=n, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    return wcss

# File path and constants
file_path = 'C:\\Users\\Revathi Nanubala\\Downloads\\minute_weather\\minute_weather.csv'
chunk_size = 50000
max_k = 10
poly_degree = 2

# Data normalization
scaler = StandardScaler()

# Reading and preprocessing data
sample_data = read_weather_data(file_path, nrows=100000, usecols=['avg_wind_direction', 'avg_wind_speed'])
sample_normalized = scaler.fit_transform(sample_data.dropna())

# WCSS calculation for Elbow Method
wcss = calculate_wcss(sample_normalized, max_k)

# Optimal number of clusters determination
n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(sample_normalized)
labels = kmeans.labels_
silhouette_avg = silhouette_score(sample_normalized, labels)

# Elbow curve plotting
plt.figure(figsize=(12, 7))
plt.plot(range(1, max_k + 1), wcss, marker='o', linestyle='-', color='gray', label='WCSS')
plt.axvline(x=n_clusters, color='red', linestyle='--', label=f'Optimal clusters: {n_clusters}')
plt.text(n_clusters, max(wcss)/2, f'Silhouette: {silhouette_avg:.2f}', color='red')
plt.title('Elbow Curve')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.legend()
plt.show()

#Processing and clustering data in chunks
all_clusters = []
for chunk in pd.read_csv(file_path, usecols=['avg_wind_direction', 'avg_wind_speed', 'hpwren_timestamp'], chunksize=chunk_size):
    chunk = chunk.dropna()
    chunk_normalized = scaler.fit_transform(chunk[['avg_wind_direction', 'avg_wind_speed']])
    chunk['cluster'] = kmeans.fit_predict(chunk_normalized)
    all_clusters.append(chunk)

#Combining chunked data
df_combined = pd.concat(all_clusters)

#Cluster centers calculation
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
sorted_centers = cluster_centers[cluster_centers[:, 0].argsort()]

#Polynomial fitting
coefficients = np.polyfit(df_combined['avg_wind_direction'], df_combined['avg_wind_speed'], poly_degree)
sorted_directions = np.sort(df_combined['avg_wind_direction'])
polynomial = np.polyval(coefficients, sorted_directions)

#Error range calculation
lower_bound, upper_bound = err_ranges(sorted_directions, df_combined['avg_wind_speed'], coefficients)

#Curve fit and cluster center plotting
plt.figure(figsize=(12, 7))
plt.plot(sorted_directions, polynomial, color='orange', label='Polynomial Fit', linestyle='--')
plt.plot(sorted_centers[:, 0], sorted_centers[:, 1], '-o', color='green', label='Cluster Centers')
plt.xlabel('Average Wind Direction')
plt.ylabel('Average Wind Speed')
plt.title('Curve Fit and K-Means Cluster Centers')
plt.legend()
plt.show()

#Data transposition and cleaning
transposed_df = sample_data.transpose()
transposed_df.fillna(method='ffill', inplace=True)

#Prediction with uncertainty
x_to_predict = 45
predicted_wind_speed = np.polyval(coefficients, x_to_predict)
error_bounds = err_ranges(np.array([x_to_predict]), np.array([predicted_wind_speed]), coefficients)

#Plotting prediction with uncertainty
plt.figure(figsize=(12, 7))
plt.plot(sorted_directions, polynomial, color='orange', label='Polynomial Fit', linestyle='--')
plt.plot(sorted_centers[:, 0], sorted_centers[:, 1], '-o', color='green', label='Cluster Centers')
plt.fill_between(sorted_directions, lower_bound, upper_bound, color='orange', alpha=0.3, label='Confidence Interval')
plt.errorbar(x_to_predict, predicted_wind_speed, yerr=[predicted_wind_speed - error_bounds[0], error_bounds[1] - predicted_wind_speed], fmt='o', color='black', label='Prediction with Uncertainty')
plt.xlabel('Average Wind Direction')
plt.ylabel('Average Wind Speed')
plt.title('Curve Fit with Prediction and Uncertainty')
plt.legend()
plt.show()

# Histograms for Average Wind Direction and Speed Over Years
df_combined['year'] = pd.to_datetime(df_combined['hpwren_timestamp'], format='%Y-%m-%d %H:%M:%S').dt.year
years = [2011, 2012, 2013, 2014]

fig, axes = plt.subplots(nrows=len(years), ncols=2, figsize=(15, 12), constrained_layout=True)
fig.suptitle('Histograms of Average Wind Direction and Speed (2011-2014)', fontsize=16)

for i, year in enumerate(years):
    year_data = df_combined[df_combined['year'] == year]
    
    axes[i, 0].hist(year_data['avg_wind_direction'], bins=30, color='skyblue', edgecolor='black')
    axes[i, 0].set_title(f'Year {year} - Avg Wind Direction')
    axes[i, 0].set_xlabel('Average Wind Direction')
    axes[i, 0].set_ylabel('Frequency')
    
    axes[i, 1].hist(year_data['avg_wind_speed'], bins=30, color='lightgreen', edgecolor='black')
    axes[i, 1].set_title(f'Year {year} - Avg Wind Speed')
    axes[i, 1].set_xlabel('Average Wind Speed')
    axes[i, 1].set_ylabel('Frequency')

plt.show()

# Low Order Polynomial Fit (Quadratic)
quadratic_coefficients = np.polyfit(df_combined['avg_wind_direction'], df_combined['avg_wind_speed'], 2)
quadratic_fit = np.polyval(quadratic_coefficients, sorted_directions)

# Plotting Quadratic Fit
plt.figure(figsize=(12, 7))
plt.scatter(df_combined['avg_wind_direction'], df_combined['avg_wind_speed'], color='lightblue', label='Data', alpha=0.5)
plt.plot(sorted_directions, quadratic_fit, color='purple', label='Quadratic Fit')
plt.xlabel('Average Wind Direction')
plt.ylabel('Average Wind Speed')
plt.title('Quadratic Fit for Average Wind Speed vs. Average Wind Direction')
plt.legend()
plt.show()

