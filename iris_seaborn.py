import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
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
График 1: Pairplot (Матрица диаграмм рассеяния) для постороения парных взаимосвязей:
С помощью команды sns.pairplot() строим график и задаем следующие условия:
data – набор данных, hue — переменная в данных, которая позволяет сопоставить аспекты графика разным цветам,
palette – цветовая гамма для графика (есть заданные в самой библиотеке), diag_kind – формы графиков,
height – размер каждого из графиков.
plt.suptitle() — это функция в библиотеке Matplotlib, которая позволяет добавить второй заголовок (подзаголовок)
к графику. И выводим график командой - plt.show()
'''

sns.pairplot(data, hue='species_name', palette='Set2', diag_kind='kde', height=2)
plt.suptitle("Pairplot of Iris Dataset", y=1.02, fontsize=16)
plt.show()

'''
Далее создаем следующий график.
График 2. Boxplot - Ящик с усами
plt.figure() - задаем размер графика.
Командой sns.boxplot() создаем график и задаем параметры: data – набор данных, 
х и у  — задает набор данных, которые используются, palette – цветовая гамма для графика (есть заданные в самой библиотеке)
plt.title() - создаем заголовок и его размер, командами plt.xlabel() и plt.ylabel() создаем названия оси х и у и их размер,
и выводим график - plt.show()
'''

plt.figure(figsize=(10, 6))
sns.boxplot(x='species_name', y='petal length (cm)', data=data, palette='Set3')
plt.title("Boxplot of Petal Length by Species", fontsize=16)
plt.xlabel("Species", fontsize=12)
plt.ylabel("Petal Length (cm)", fontsize=12)
plt.show()


'''
Далее создаем следующий график.
График 3. Violinplot для сравнения вероятности распределения:
plt.figure() - задаем размер графика.
Командой sns.violinplot() создаем график и задаем параметры: data – набор данных, 
х и у  — задает набор данных, которые используются, palette – цветовая гамма для графика (есть заданные в самой библиотеке),
split – это параметр, который позволяет разделить каждый виолончель пополам. 
Это помогает лучше визуализировать различия между категориями
plt.title() - создаем заголовок и его размер, командами plt.xlabel() и plt.ylabel() создаем названия оси х и у и их размер,
и выводим график - plt.show()
'''

plt.figure(figsize=(10, 6))
sns.violinplot(x='species_name', y='sepal width (cm)', data=data, palette='muted', split=True)
plt.title("Violin Plot of Sepal Width by Species", fontsize=16)
plt.xlabel("Species", fontsize=12)
plt.ylabel("Sepal Width (cm)", fontsize=12)
plt.show()

'''
Далее создаем следующий график.
График 4. Корреляционная матрица:
plt.figure() - задаем размер графика.
Вначале необходимо построить корреляцию для heatmap, чтобы визуализировать взаимосвязи между переменными в наборе данных. 
Такая матрица помогает понять, как разные переменные тесно связаны друг с другом. Каждая ячейка в ней 
представляет коэффициент корреляции, который измеряет силу и направление связи между двумя переменными. 
correlation_matrix = data.iloc[:, :-2].corr() 
Командой sns.heatmap() создаем график и задаем параметры: correlation_matrix – набор данных, созданных при помощи 
корреляционной матрицы, аргумент annot=True, который помогает отобразить коэффициент корреляции, 
fmt – функция отображает количество цифр после запятой (больше нуля), cmap – для выбора цвета, 
cbar – добавление цветовой шкалы.  
plt.title() - создаем заголовок и его размер, командами plt.xlabel() и plt.ylabel() создаем названия оси х и у и их размер,
и выводим график - plt.show()
'''

plt.figure(figsize=(8, 6))
correlation_matrix = data.iloc[:, :-2].corr()  # Выбираем корреляцию по числовым признакам
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
plt.title("Correlation Matrix of Iris Dataset", fontsize=16)
plt.show()