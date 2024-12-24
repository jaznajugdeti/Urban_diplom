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
def seaborn_iris_pairplot(hue, palette, diag_kind, height, fontsize):
    sns.pairplot(data, hue=hue, palette=palette, diag_kind=diag_kind, height=height)
    plt.suptitle("Pairplot of Iris Dataset", y=1.02, fontsize=fontsize)
    plt.show()
seaborn_iris_pairplot('species_name','Set2', 'kde', 2, 16 )

'''
Далее создаем следующий график.
График 2. Boxplot - Ящик с усами
plt.figure() - задаем размер графика.
Командой sns.boxplot() создаем график и задаем параметры: data – набор данных, 
х и у  — задает набор данных, которые используются, palette – цветовая гамма для графика (есть заданные в самой библиотеке)
plt.title() - создаем заголовок и его размер, командами plt.xlabel() и plt.ylabel() создаем названия оси х и у и их размер,
и выводим график - plt.show()
'''
def seaborn_iris_boxplot(f1, f2, palette, fontsize):
    plt.figure(figsize=(f1, f2))
    sns.boxplot(x='species_name', y='petal length (cm)', data=data, palette=palette)
    plt.title("Boxplot of Petal Length by Species", fontsize=fontsize+4)
    plt.xlabel("Species", fontsize=fontsize)
    plt.ylabel("Petal Length (cm)", fontsize=fontsize)
    plt.show()

seaborn_iris_boxplot(10,6, 'Set3', 12)


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
def seaborn_iris_violinplot(f1, f2, x, y, palette, fontsize):
    plt.figure(figsize=(f1, f2))
    sns.violinplot(x=x, y=y, data=data, palette=palette, split=True)
    plt.title("Violin Plot of Sepal Width by Species", fontsize=fontsize+4)
    plt.xlabel("Species", fontsize=fontsize)
    plt.ylabel("Sepal Width (cm)", fontsize=fontsize)
    plt.show()
seaborn_iris_violinplot(10,6, 'species_name', 'sepal width (cm)','muted', 12)


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
def seaborn_iris_heatmap(f1, f2, cmap, fmt, cbar, fontsize):
    plt.figure(figsize=(f1, f2))
    correlation_matrix = data.iloc[:, :-2].corr()  # Выбираем корреляцию по числовым признакам
    sns.heatmap(correlation_matrix, annot=True, cmap=cmap, fmt=fmt, cbar=cbar)
    plt.title("Correlation Matrix of Iris Dataset", fontsize=fontsize)
    plt.show()
seaborn_iris_heatmap(8,6, 'coolwarm', ".2f",True, 16)