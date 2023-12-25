import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Загрузка данных
customers_df = pd.read_csv('customers.csv')

# Вывод первых и последних 10 строк
print(customers_df.head(10))
print(customers_df.tail(10))

# Статистика по значениям признаков
print(customers_df.describe())

# Определение семейного положения клиентов
marital_status_counts = customers_df['Genre'].value_counts()
plt.pie(marital_status_counts, labels=marital_status_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Пол')
plt.show()

# Исследование связи между возрастом и годовым доходом клиентов
sns.scatterplot(x='Age', y='Annual Income (k$)', data=customers_df)
plt.title('Связь между возрастом и годовым доходом')
plt.show()

# # Анализ профессий клиентов
# plt.figure(figsize=(12, 6))
# sns.countplot(x='Profession', data=customers_df, order=customers_df['Profession'].value_counts().index)
# plt.xticks(rotation=45, ha='right')
# plt.title('Распределение клиентов по профессиям')
# plt.show()

# Анализ годового дохода клиентов
plt.figure(figsize=(12, 6))
sns.histplot(customers_df['Annual Income (k$)'], bins=20, kde=True)
plt.title('Гистограмма годового дохода клиентов')
plt.show()

# Анализ возрастного распределения в зависимости от пола клиента
plt.figure(figsize=(10, 6))
sns.boxplot(x='Genre', y='Age', data=customers_df)
plt.title('Возрастное распределение в зависимости от пола')
plt.show()

# Подготовка данных для кластеризации
X = customers_df[['Age', 'Annual Income (k$)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
customers_df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

dbscan = DBSCAN(eps=0.5, min_samples=5)
customers_df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

# Определение оптимального количества кластеров с использованием метода локтя (KMeans)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Метод локтя для определения оптимального числа кластеров')
plt.xlabel('Количество кластеров')
plt.ylabel('Сумма квадратов расстояний')
plt.show()

# Определение оптимального количества кластеров с использованием silhouette_score (KMeans)
silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))

optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
print(f'Оптимальное количество кластеров (KMeans): {optimal_clusters}')

# Визуализация результатов кластеризации
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x='Age', y='Annual Income (k$)', hue='KMeans_Cluster', data=customers_df, palette='viridis')
plt.title('KMeans Clustering')

plt.subplot(1, 2, 2)
sns.scatterplot(x='Age', y='Annual Income (k$)', hue='DBSCAN_Cluster', data=customers_df, palette='viridis')
plt.title('DBSCAN Clustering')

plt.show()
