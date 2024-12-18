import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

# Шаг 1: Загружаем данные из библиотеки Wine
wine = load_wine()
data = pd.DataFrame(wine.data, columns=wine.feature_names)
data['target'] = wine.target
data['target_name'] = data['target'].apply(lambda x: wine.target_names[x])

# Шаг 2: Строим графики с помощью Seaborn
# График 1: Pairplot (Матрица диаграмм рассеяния) для постороения парных взаимосвязей:
# # задаем цветовую палитру, высоту, вид, размер
sns.pairplot(data,
             hue="target_name",
             palette="Set2",
             diag_kind="kde",
             corner=True,
             height=2.5)
plt.suptitle("Pairplot of Wine Dataset", y=1.02, fontsize=16)
# показать график
plt.show()

#  График 2. Boxplot - Ящик с усами (для анализа распределения алкоголя)
# # задаем цветовую палитру, высоту, вид, размер, что отображаться будет по х и у, название осей
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x="target_name", y="alcohol", palette="Set3")
plt.title("Boxplot of Alcohol Content by Wine Type", fontsize=16)
plt.xlabel("Wine Type", fontsize=12)
plt.ylabel("Alcohol Content", fontsize=12)
# создаем сетку показываем график
plt.grid(axis="y", linestyle="--", alpha=0.7)
#  настройка промежутка между ящиками
plt.tight_layout()
# показать график
plt.show()


# График 3. Violinplot для сравнения вероятности распределения: (для анализа яблочной кислоты)
# задаем цветовую палитру, высоту, вид, размер, что отображаться будет по х и у, название осей
plt.figure(figsize=(10, 6))
sns.violinplot(data=data, x="target_name", y="malic_acid", palette="muted", split=True)
plt.title("Violin Plot of Malic Acid Content by Wine Type", fontsize=16)
plt.xlabel("Wine Type", fontsize=12)
plt.ylabel("Malic Acid Content", fontsize=12)
# создаем сетку, показываем график, автоматическая настройка промежутка
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# График 4. Корреляционная матрица:
# задаем цветовую палитру, палитру, формат отображения данных
plt.figure(figsize=(10, 8))
correlation_matrix = data.iloc[:, :-2].corr()  # Корреляция между числовыми признаками
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
plt.title("Correlation Matrix of Wine Dataset", fontsize=16)
# показываем график, автоматическая настройка промежутка
plt.tight_layout()
plt.show()

# График 5: Strip Plot - для распределения многих индивидуальных одномерных значений (для флавоноидов)
# задаем цветовую палитру, высоту, вид, размер, что отображаться будет по х и у, название осей
plt.figure(figsize=(10, 6))
sns.stripplot(data=data, x="target_name", y="flavanoids", jitter=True, palette="Set1", alpha=0.7)
plt.title("Strip Plot of Flavanoids by Wine Type", fontsize=16)
plt.xlabel("Wine Type", fontsize=12)
plt.ylabel("Flavanoids", fontsize=12)
# создаем сетку, показываем график, автоматическая настройка промежутка
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()