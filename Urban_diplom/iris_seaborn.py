import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Шаг 1: Загружаем данные из библиотеки Iris
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target
data['species_name'] = data['species'].apply(lambda x: iris.target_names[x])

# Шаг 2: Строим графики с помощью Seaborn

# График 1. Pairplot для постороения парных взаимосвязей:
# задаем цветовую палитру, высоту, вид, размер
sns.pairplot(data, hue='species_name', palette='Set2', diag_kind='kde', height=2)
plt.suptitle("Pairplot of Iris Dataset", y=1.02, fontsize=16)
# показать график
plt.show()

# График 2. Boxplot - Ящик с усами
# задаем цветовую палитру, высоту, вид, размер, что отображаться будет по х и у, название осей
plt.figure(figsize=(10, 6))
sns.boxplot(x='species_name', y='petal length (cm)', data=data, palette='Set3')
plt.title("Boxplot of Petal Length by Species", fontsize=16)
plt.xlabel("Species", fontsize=12)
plt.ylabel("Petal Length (cm)", fontsize=12)
# показываем график
plt.show()

# График 3. Violinplot для сравнения вероятности распределения:
# задаем цветовую палитру, высоту, вид, размер, что отображаться будет по х и у, название осей
plt.figure(figsize=(10, 6))
sns.violinplot(x='species_name', y='sepal width (cm)', data=data, palette='muted', split=True)
plt.title("Violin Plot of Sepal Width by Species", fontsize=16)
plt.xlabel("Species", fontsize=12)
plt.ylabel("Sepal Width (cm)", fontsize=12)
# показываем график
plt.show()

# График 4. Корреляционная матрица:
# задаем цветовую палитру, палитру, формат отображения данных
plt.figure(figsize=(8, 6))
correlation_matrix = data.iloc[:, :-2].corr()  # Выбираем корреляцию по числовым признакам
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
plt.title("Correlation Matrix of Iris Dataset", fontsize=16)
# показываем график
plt.show()