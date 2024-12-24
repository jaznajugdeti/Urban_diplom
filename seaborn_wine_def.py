import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

''' 
Скачиваем необходимые библиотеки.
Далее необходимо провести ряд изменений в датасете для удобства:
wine = load_wine() - скачиванем датасет;
data['target'] = wine.target - добавляем целевую переменную;
data = pd.DataFrame(wine.data, columns=wine.feature_names) — это превращение датасета о вине в датафрейм
с использованием библиотеки pandas.
data['target_name'] = data['target'].apply(lambda x: wine.target_names[x]) - Это код для добавления нового столбца 
с именами целей для лучшей читаемости. Он использует метод apply() для применения лямбда-функции к столбцу «target» и 
получения значений из объекта wine.target_names по этому столбцу;

'''

wine = load_wine()
data = pd.DataFrame(wine.data, columns=wine.feature_names)
data['target'] = wine.target
data['target_name'] = data['target'].apply(lambda x: wine.target_names[x])

''' 
Далее мы начинаем строить графики.
График 1: Pairplot (Матрица диаграмм рассеяния) для постороения парных взаимосвязей:
С помощью команды sns.pairplot() строим график и задаем следующие условия:
data – набор данных, hue — переменная в данных, которая позволяет сопоставить аспекты графика разным цветам,
palette – цветовая гамма для графика (есть заданные в самой библиотеке), diag_kind – формы графиков,
height – размер каждого из графиков, а также  corner – для задачи дополнительных осей.
plt.suptitle() — это функция в библиотеке Matplotlib, которая позволяет добавить второй заголовок (подзаголовок)
к графику. И выводим график командой - plt.show()
'''
def seaborn_wine_pairplot(hue, palette, diag_kind, height, fontsize):
    sns.pairplot(data,
             hue=hue,
             palette=palette,
             diag_kind=diag_kind,
             corner=True,
             height=height)
    plt.suptitle("Pairplot of Wine Dataset", y=1.02, fontsize=fontsize)
    plt.show()

seaborn_wine_pairplot("target_name",'Set2', 'kde', 2.5, 16 )

'''
Далее создаем следующий график.
График 2. Boxplot - Ящик с усами (для анализа распределения алкоголя)
plt.figure() - задаем размер графика.
Командой sns.boxplot() создаем график и задаем параметры: data – набор данных, 
х и у  — задает набор данных, которые используются, palette – цветовая гамма для графика (есть заданные в самой библиотеке)
plt.title() - создаем заголовок и его размер, командами plt.xlabel() и plt.ylabel() создаем названия оси х и у и их размер,
plt.grid() -  создаем сетку, по какой оси откладываем ее прозрачность (alpha) и форму линии (linestyle).
plt.tight_layout() —  оптимизация расположения элементов графика, и выводим график - plt.show()
'''

def seaborn_wine_boxplot(f1, f2, x, y, palette, fontsize, alpha):
    plt.figure(figsize=(f1, f2))
    sns.boxplot(data=data, x=x, y=y, palette=palette)
    plt.title("Boxplot of Alcohol Content by Wine Type", fontsize=fontsize+4)
    plt.xlabel("Wine Type", fontsize=fontsize)
    plt.ylabel("Alcohol Content", fontsize=fontsize)
    plt.grid(axis="y", linestyle="--", alpha=alpha)
    plt.tight_layout()
    plt.show()
seaborn_wine_boxplot(10,6, "target_name", "alcohol", "Set3",  12, 0.7)

'''
Далее создаем следующий график.
График 3. Violinplot для сравнения вероятности распределения: (для анализа яблочной кислоты)
plt.figure() - задаем размер графика.
Командой sns.violinplot() создаем график и задаем параметры: data – набор данных, 
х и у  — задает набор данных, которые используются, palette – цветовая гамма для графика (есть заданные в самой библиотеке),
split – это параметр, который позволяет разделить каждый виолончель пополам. 
Это помогает лучше визуализировать различия между категориями
plt.title() - создаем заголовок и его размер, командами plt.xlabel() и plt.ylabel() создаем названия оси х и у и их размер,
plt.grid() -  создаем сетку, по какой оси откладываем ее прозрачность (alpha) и форму линии (linestyle).
plt.tight_layout() —  оптимизация расположения элементов графика, и выводим график - plt.show()
'''

def seaborn_wine_violinplot(f1, f2, x, y, palette, fontsize, alpha):
    plt.figure(figsize=(f1, f2))
    sns.violinplot(data=data, x=x, y=y, palette="muted", split=True)
    plt.title("Violin Plot of Malic Acid Content by Wine Type", fontsize=fontsize+4)
    plt.xlabel("Wine Type", fontsize=fontsize)
    plt.ylabel("Malic Acid Content", fontsize=fontsize)
    plt.grid(axis="y", linestyle="--", alpha=alpha)
    plt.tight_layout()
    plt.show()
seaborn_wine_violinplot(10,6, "target_name", "malic_acid",'muted', 12, 0.7)

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
plt.grid() -  создаем сетку, по какой оси откладываем ее прозрачность (alpha) и форму линии (linestyle).
plt.tight_layout() —  оптимизация расположения элементов графика, и выводим график - plt.show()
'''

def seaborn_wine_heatmap(f1, f2, cmap, fmt, cbar, fontsize):
    plt.figure(figsize=(f1, f2))
    correlation_matrix = data.iloc[:, :-2].corr()  # Корреляция между числовыми признаками
    sns.heatmap(correlation_matrix, annot=True, cmap=cmap, fmt=fmt, cbar=cbar)
    plt.title("Correlation Matrix of Wine Dataset", fontsize=fontsize)
    plt.tight_layout()
    plt.show()
seaborn_wine_heatmap(10,8, "coolwarm", ".2f",True, 16)

'''
Далее создаем следующий график.
График 5: Strip Plot - для распределения многих индивидуальных одномерных значений (для флавоноидов)
plt.figure() - задаем размер графика.
Командой sns.stripplot() создаем график и задаем параметры: data – набор данных, х и у  — задает набор данных, 
которые используются, jitter – величина дрожания (только вдоль категориальной оси) для приложения. 
Этот может быть полезно, когда у вас много точек и они перекрываются, palette – цветовая гамма для графика 
(есть заданные в самой библиотеке), alpha – прозрачность.
plt.title() - создаем заголовок и его размер, командами plt.xlabel() и plt.ylabel() создаем названия оси х и у и их размер,
plt.grid() -  создаем сетку, по какой оси откладываем ее прозрачность (alpha) и форму линии (linestyle).
plt.tight_layout() —  оптимизация расположения элементов графика, и выводим график - plt.show()
'''

def seaborn_wine_stripplot(f1, f2, x, y, palette, fontsize, alpha):
    plt.figure(figsize=(f1, f2))
    sns.stripplot(data=data, x=x, y=y, jitter=True, palette=palette, alpha=alpha)
    plt.title("Strip Plot of Flavanoids by Wine Type", fontsize=fontsize+4)
    plt.xlabel("Wine Type", fontsize=fontsize)
    plt.ylabel("Flavanoids", fontsize=fontsize)
    plt.grid(axis="y", linestyle="--", alpha=alpha)
    plt.tight_layout()
    plt.show()
seaborn_wine_stripplot(10,6, "target_name", "flavanoids","Set1", 16, 0.7)