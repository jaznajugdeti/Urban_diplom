import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris


''' 
Скачиваем необходимые библиотеки.
Далее необходимо провести ряд изменений в датасете для удобства:
iris = load_iris() - скачиванем датасет;
data['species'] = iris.target- добавляем целевую переменную;
data = pd.DataFrame(data=iris.data, columns=iris.feature_names) — это превращение датасета о цветах в датафрейм
с использованием библиотеки pandas.
data['species_name'] = data['species'].apply(lambda x: iris.target_names[x]) - Этот код добавляет столбец с названиями 
видов ирисов на основе целевой переменной (iris.target);

'''

iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target
data['species_name'] = data['species'].apply(lambda x: iris.target_names[x])

''' 
Далее мы начинаем строить графики.
ГГрафик 1. Гистограммы лепестков.
plt.figure() - задаем размер графика.
С помощью команды plt.hist() строим график и задаем следующие условия:
data – набор данных, bins – количество столбцов, alpha – прозрачность, data – сет данных, color - цвет, 
label -  позволяет указать строку или None для отображения меток.
plt.title() - создаем заголовок и его размер, командами plt.xlabel() и plt.ylabel() создаем названия оси х и у и их размер,
plt.grid() -  создаем сетку, по какой оси откладываем ее прозрачность (alpha) и форму линии (linestyle). plt.legend() -
создаем легенду.
plt.tight_layout() —  оптимизация расположения элементов графика, и выводим график - plt.show().
'''

plt.figure(figsize=(10, 6))
plt.hist(data['sepal length (cm)'], bins=15, alpha=0.7, label='Sepal Length', color='blue')
plt.hist(data['petal length (cm)'], bins=15, alpha=0.7, label='Petal Length', color='orange')
plt.title("Distribution of Sepal and Petal Lengths", fontsize=16)
plt.xlabel("Length (cm)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

''' 
Строим следующий график.
График 2. Диаграмма рассеяния:
plt.figure() - задаем размер графика.
colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'} - задаем изначально цвета для каждой из группы.
for species, group in data.groupby('species_name'): - разбиваем данные по параметру species_name и на основе растасовки 
данных строим график командой plt.scatter(), где задаются два массива данных  точками данных 
group['sepal length (cm)'], group['sepal width (cm)'] — один для оси x, другой для оси y,
и отображает их как отдельные точки на графике. А также alpha – прозрачность, color - цвет, 
label -  позволяет указать строку или None для отображения меток.
plt.title() - создаем заголовок и его размер, командами plt.xlabel() и plt.ylabel() создаем названия оси х и у и их размер,
plt.grid() -  создаем сетку, по какой оси откладываем ее прозрачность (alpha) и форму линии (linestyle). plt.legend() -
создаем легенду.
plt.tight_layout() —  оптимизация расположения элементов графика, и выводим график - plt.show().
'''

plt.figure(figsize=(10, 6))
colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
for species, group in data.groupby('species_name'):
    plt.scatter(group['sepal length (cm)'], group['sepal width (cm)'],
                label=species, color=colors[species], alpha=0.7)
plt.title("Sepal Length vs Sepal Width", fontsize=16)
plt.xlabel("Sepal Length (cm)", fontsize=12)
plt.ylabel("Sepal Width (cm)", fontsize=12)
plt.legend(title="Species")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

''' 
Далее мы начинаем строить графики.
График 3. Круговая диаграмма:
Species_counts = data['species_name'].value_counts() — это подсчёт количества уникальных значений 
в столбце «species» в фрейме данных 
plt.figure() - задаем размер графика.
Строим график командой plt.pie(),  выбираем данные для анализа - species_counts, labels=species_counts.index — это способ 
получить метки для визуализации иерархических данных с помощью вложенных круговых диаграмм, 
autopct – для одобрения показывать процент, startangle  - задает место (градус) откуда начианем рисовать, 
colors – задает цвета.
plt.title() - создаем заголовок и его размер, командами и выводим график - plt.show().
'''

species_counts = data['species_name'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(species_counts, labels=species_counts.index, autopct='%1.1f%%', startangle=90, colors=['red', 'green', 'blue'])
plt.title("Distribution of Iris Species", fontsize=16)
plt.show()