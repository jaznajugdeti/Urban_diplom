import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

# Шаг 1: Загружаем данные из библиотеки Iris
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target

# Добавляем название видов для удобства
data['species_name'] = data['species'].apply(lambda x: iris.target_names[x])

# Шаг 2: Строим графики на базе датасета Iris
# График 1. Гистограммы лепестков:
plt.figure(figsize=(10, 6)) # создаем размер графика
# выбираем данные,  цвет и прозрачность, и количество столбов
plt.hist(data['sepal length (cm)'], bins=15, alpha=0.7, label='Sepal Length', color='blue')
plt.hist(data['petal length (cm)'], bins=15, alpha=0.7, label='Petal Length', color='orange')
# задаем название графика
plt.title("Distribution of Sepal and Petal Lengths", fontsize=16)
# задем названия осей,
plt.xlabel("Length (cm)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend()
# создаем сетку, автоматическая настройка промежутка
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# График 2. Диаграмма рассеяния:
plt.figure(figsize=(10, 6)) # создаем размер графика
# задаем цвета для групп
colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
# получаем нужные нам характеристики
for species, group in data.groupby('species_name'):
    plt.scatter(group['sepal length (cm)'], group['sepal width (cm)'],
                label=species, color=colors[species], alpha=0.7)
# задаем название графика и осей
plt.title("Sepal Length vs Sepal Width", fontsize=16)
plt.xlabel("Sepal Length (cm)", fontsize=12)
plt.ylabel("Sepal Width (cm)", fontsize=12)
# содаем легенду
plt.legend(title="Species")
# создаем сетку, автоматическая настройка промежутка
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
# выводим график
plt.show()

# График 3. Круговая диаграмма:
species_counts = data['species_name'].value_counts()
# создаем размер графика
plt.figure(figsize=(8, 8))
# задаем название, что хотим показывать процент, откуда начианем рисовать и цвета
plt.pie(species_counts, labels=species_counts.index, autopct='%1.1f%%', startangle=90, colors=['red', 'green', 'blue'])
plt.title("Distribution of Iris Species", fontsize=16)
plt.show()