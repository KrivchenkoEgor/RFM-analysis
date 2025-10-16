# -*- coding: utf-8 -*-
"""
RFM-анализ и кластеризация клиентов по KMeans на основе датасета OnlineRetail.csv закончен
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Устанавливаем стиль для графиков
sns.set(style="whitegrid")

# 1. Загрузка и предварительная обработка данных
print("1. Загрузка данных...")
df = pd.read_csv('/Users/egorkrivchenko/PycharmProjects/scikit-learn/OnlineRetail.csv', encoding='ISO-8859-1')

# Удаляем строки с пропущенными CustomerID (без них RFM невозможен)
df = df.dropna(subset=['CustomerID'])

# Удаляем отмененные заказы (InvoiceNo начинается с 'C') и некорректные количества
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] >= 0]

# Преобразуем CustomerID в целое число для удобства
df['CustomerID'] = df['CustomerID'].astype(int)

# Преобразуем дату
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d-%m-%Y %H:%M')

print(f"Данные загружены. Количество строк после очистки: {len(df)}")

# 2. Создание RFM-метрик
print("2. Создание RFM-метрик...")

# Определяем "сегодняшнюю" дату как последнюю дату в датасете + 1 день
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# Рассчитываем общую сумму для каждой строки
df['TotalSum'] = df['Quantity'] * df['UnitPrice']

# Группируем по CustomerID
rfm_table = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days, # Recency
    'InvoiceNo': 'nunique',                                  # Frequency
    'TotalSum': 'sum'                                        # Monetary
}).reset_index()

# Переименовываем колонки
rfm_table.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# Удаляем клиентов с Monetary = 0 (могут быть из-за UnitPrice=0)
rfm_table = rfm_table[rfm_table['Monetary'] > 0]

print(f"RFM-таблица создана. Количество клиентов: {len(rfm_table)}")

# 3. Нормализация данных
print("3. Нормализация данных...")
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_table[['Recency', 'Frequency', 'Monetary']])
rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'])

# 4. Определение оптимального числа кластеров
print("4. Поиск оптимального числа кластеров...")

# Метод локтя и Silhouette Score
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled_df)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(rfm_scaled_df, kmeans.labels_))

# Визуализация
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Количество кластеров (k)')
ax1.set_ylabel('Инерция (Inertia)', color=color)
ax1.plot(K_range, inertias, 'o-', color=color, label='Инерция')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Silhouette Score', color=color)
ax2.plot(K_range, silhouette_scores, 's--', color=color, label='Silhouette Score')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Метод локтя и Silhouette Score для выбора k')
plt.show()

# Выводим таблицу с метриками для удобства
results_df = pd.DataFrame({
    'k': K_range,
    'Inertia': inertias,
    'Silhouette_Score': silhouette_scores
})
print("\nМетрики для разных значений k:")
print(results_df.to_string(index=False))

# Пользователь выбирает k (обычно k=4 или k=5 для RFM)
k_optimal = int(input("\nВведите оптимальное количество кластеров (k) на основе графиков: "))

# 5. Кластеризация
print(f"5. Кластеризация с k={k_optimal}...")
kmeans_final = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
rfm_table['Cluster'] = kmeans_final.fit_predict(rfm_scaled_df)

# === НОВЫЙ БЛОК КОДА ===
# 6. Добавление колонки с размером кластера
print("6. Добавление информации о размере кластера...")
cluster_sizes = rfm_table['Cluster'].value_counts().to_dict()
rfm_table['Cluster_Size'] = rfm_table['Cluster'].map(cluster_sizes)
# === КОНЕЦ НОВОГО БЛОКА ===

# 7. Анализ кластеров
print("7. Анализ кластеров...")
cluster_summary = rfm_table.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count']
}).round(2)

cluster_summary.columns = ['Recency_mean', 'Frequency_mean', 'Monetary_mean', 'Count']
cluster_summary = cluster_summary.sort_values('Monetary_mean', ascending=False)

print("\nСводка по кластерам:")
print(cluster_summary)

# Визуализация кластеров (например, Recency vs Monetary)
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=rfm_table,
    x='Recency',
    y='Monetary',
    hue='Cluster',
    palette='Set1',
    alpha=0.7
)
plt.title('Кластеризация клиентов (Recency vs Monetary)')
plt.xlabel('Recency (дни)')
plt.ylabel('Monetary (общая сумма)')
plt.legend(title='Кластер')
plt.show()

# Сохранение результатов
output_path = '/Users/egorkrivchenko/PycharmProjects/scikit-learn/RFM_Clusters.csv'
rfm_table.to_csv(output_path, index=False)
print(f"\nРезультаты сохранены в файл: {output_path}")

# Пример интерпретации (можно адаптировать под ваши данные)
print("\n--- Пример интерпретации кластеров ---")
print("Кластер 0 (самый высокий Monetary): 'Лучшие клиенты' - высокая частота, недавние покупки.")
print("Кластер с высоким Recency и низким Monetary: 'Потерянные клиенты'.")
print("Кластер с высокой Frequency, но низким Monetary: 'Бюджетные клиенты'.")