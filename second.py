import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 0. Описание задачи
print("Задача: классификация ноутбуков по ценовому диапазону на основе характеристик.")

# 1. Чтение данных
file_path = "Laptop-Price.csv"
df = pd.read_csv(file_path)
print(f"Размер данных: {df.shape}")

# 2. Визуализация и основные характеристики
print("\nСтатистические характеристики данных:")
print(df.describe())

# Оставляем только числовые столбцы перед построением корреляционной матрицы
numeric_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Корреляционная матрица")
plt.show()

# 3. Обработка пропущенных значений
df = df.drop(columns=["Unnamed: 16", "Product", "ScreenResolution", "Cpu Model", "Gpu Model"], errors='ignore')

print(f"После удаления ненужного столбца: {df.shape}")

# 4. Обработка категориальных признаков
categorical_columns = ["Company", "TypeName", "OpSys", "Cpu Brand", "Gpu Brand"]
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Преобразование 'Ram' и 'Cpu Rate' в числовые значения
df["Ram"] = df["Ram"].str.replace("GB", "").astype(int)
df["Cpu Rate"] = df["Cpu Rate"].str.replace("GHz", "").astype(float)

# 5. Нормализация данных
scaler = StandardScaler()
numeric_columns = ["Inches", "Ram", "Cpu Rate", "SSD", "HDD", "Flash Storage", "Hybrid"]
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# 6. Разбиение данных на обучающую и тестовую выборки
X = df.drop(columns=["Price_euros"])
y = pd.qcut(df["Price_euros"], q=3, labels=[0, 1, 2])  # Разбиваем цены на три категории
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Запуск классификатора (KNN)
k_values = range(1, 21)
accuracy_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

best_k = k_values[np.argmax(accuracy_scores)]
print(f"Лучшее значение k: {best_k}")
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred_train = knn.predict(X_train)
y_pred_test = knn.predict(X_test)

# 8. Оценка качества модели
print("\nМатрица рассогласования (Test):")
print(confusion_matrix(y_test, y_pred_test))
print("\nКлассификационный отчет:")
print(classification_report(y_test, y_pred_test))
print(f"Точность модели: {accuracy_score(y_test, y_pred_test):.4f}")

# Визуализация ошибок
plt.figure(figsize=(10, 5))
plt.plot(k_values, accuracy_scores, marker='o', linestyle='-')
plt.xlabel("Число соседей (k)")
plt.ylabel("Точность")
plt.title("Выбор оптимального k для KNN")
plt.show()
