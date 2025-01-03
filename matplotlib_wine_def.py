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
График 1. Гистограммы (Alcohol Content Distribution).
plt.figure() - задаем размер графика.
for target_name, group in data.groupby('target_name') - разделяет фрейм данных на группы на основе заданных
критериев (target_name).
С помощью команды plt.hist() строим график и задаем следующие условия:
group['alcohol'] - это функция для построения гистограммы по столбцу «alcohol» – набор данных, 
bins – количество столбцов, alpha – прозрачность, color - цвет, 
label -  позволяет указать строку или None для отображения меток.
plt.title() - создаем заголовок и его размер, командами plt.xlabel() и plt.ylabel() создаем названия оси х и у и их размер,
plt.grid() -  создаем сетку, по какой оси откладываем ее прозрачность (alpha) и форму линии (linestyle). plt.legend() -
создаем легенду.
plt.tight_layout() —  оптимизация расположения элементов графика, и выводим график - plt.show().
'''
def matplotlib_wine_hist(f1, f2, fontsize, bins, alpha):
    plt.figure(figsize=(f1, f2))
    for target_name, group in data.groupby('target_name'):
        plt.hist(group['alcohol'], bins=bins, alpha=alpha, label=target_name)
    plt.title("Alcohol Content Distribution by Wine Type", fontsize=fontsize+4)
    plt.xlabel("Alcohol Content (%)", fontsize=fontsize)
    plt.ylabel("Frequency", fontsize=fontsize)
    plt.legend(title="Wine Type")
    plt.grid(axis='y', linestyle='--', alpha=alpha)
    plt.tight_layout()
    plt.show()
matplotlib_wine_hist(10, 6, 12, 15, 0.7)

''' 
Строим следующий график.
График 2. Диаграмма рассеяния (Flavanoids vs Color Intensity):
plt.figure() - задаем размер графика.
colors = ['red', 'green', 'blue'] - задаем изначально цвета для групп.

Функция enumerate() создаёт объект-генератор, который генерирует кортежи, состоящие из двух элементов 
— индекса элемента и самого элемента, subset = data[data['target'] == i] - Эта конструкция фильтрует DataFrame, 
возвращая только те строки, где значение в столбце target равно i, дальше с помощью команды plt.scatter() строим график,
где subset[] - набор отсортированных данных,  alpha – прозрачность, color - цвет, 
label -  позволяет указать строку или None для отображения меток.
plt.title() - создаем заголовок и его размер, командами plt.xlabel() и plt.ylabel() создаем названия оси х и у и их размер,
plt.grid() -  создаем сетку, по какой оси откладываем ее прозрачность (alpha) и форму линии (linestyle). plt.legend() -
создаем легенду.
plt.tight_layout() —  оптимизация расположения элементов графика, и выводим график - plt.show().
'''
def matplotlib_wine_scatter(f1, f2, fontsize, alpha, title):
    plt.figure(figsize=(f1, f2))
    colors = ['red', 'green', 'blue']
    for i, target_name in enumerate(wine.target_names):
        subset = data[data['target'] == i]
        plt.scatter(subset['flavanoids'], subset['color_intensity'],
                    label=target_name, color=colors[i], alpha=alpha)
    plt.title("Flavanoids vs Color Intensity", fontsize=fontsize+4)
    plt.xlabel("Flavanoids", fontsize=fontsize)
    plt.ylabel("Color Intensity", fontsize=fontsize)
    plt.legend(title=title)
    plt.grid(True, linestyle='--', alpha=alpha)
    plt.tight_layout()
    plt.show()
matplotlib_wine_scatter(10,6, 12, 0.7, "Wine Type")
''' 
Строим следующий график.
График 3. Boxplot - ящик с усами: (Malic Acid by Wine Type)
plt.figure() - задаем размер графика.
data.boxplot() — это метод в Pandas, который создаёт бокс-плот (также известный как бокс-и-вискир-плот) 
визуализации из числовых данных в столбцах DataFrame, где column – название данных, которые откладываются, 
by – группировка данных по критерию, grid – сетка, patch_artist – заполняем цветом сами ящики, 
boxprops=dict(facecolor='lightblue') – метод назначения цвета для всех.
plt.title() - создаем заголовок и его размер, командами plt.xlabel() и plt.ylabel() создаем названия оси х и у и их размер,
plt.tight_layout() —  оптимизация расположения элементов графика, и выводим график - plt.show().
'''

def matplotlib_wine_boxplot(f1, f2, fontsize,column, by):
    plt.figure(figsize=(f1, f2))
    data.boxplot(column=column, by=by, grid=False, patch_artist=True,
                 boxprops=dict(facecolor='lightblue'))
    plt.title("Boxplot of Malic Acid Content by Wine Type", fontsize=fontsize+4)
    plt.suptitle("")  # Убираем лишний заголовок
    plt.xlabel("Wine Type", fontsize=fontsize)
    plt.ylabel("Malic Acid Content", fontsize=fontsize)
    plt.tight_layout()
    plt.show()

matplotlib_wine_boxplot(10,6, 12,  'malic_acid', 'target_name')

''' 
Строим следующий график.
График 4. Гистограмма (Total Phenols Distribution):
plt.figure() - задаем размер графика.
С помощью команды plt.hist() строим график и задаем следующие условия:
data - набор данных по столбцу 'total_phenols', 
bins – количество столбцов, alpha – прозрачность, color - цвет, 
edgecolor — это параметр, который задаёт цвет границы столбцов.
plt.title() - создаем заголовок и его размер, командами plt.xlabel() и plt.ylabel() создаем названия оси х и у и их размер,
plt.grid() -  создаем сетку, по какой оси откладываем ее прозрачность (alpha) и форму линии (linestyle). plt.legend() -
создаем легенду.
plt.tight_layout() —  оптимизация расположения элементов графика, и выводим график - plt.show().
'''
def matplotlib_wine_hist2(f1, f2, fontsize, bins, alpha, color, edgecolor):
    plt.figure(figsize=(f1, f2))
    plt.hist(data['total_phenols'], bins=bins, color=color, edgecolor=edgecolor, alpha=alpha)
    plt.title("Distribution of Total Phenols", fontsize=fontsize+4)
    plt.xlabel("Total Phenols", fontsize=fontsize)
    plt.ylabel("Frequency", fontsize=fontsize)
    plt.grid(axis='y', linestyle='--', alpha=alpha)
    plt.tight_layout()
    plt.show()
matplotlib_wine_hist2(10, 6, 12, 20, 0.7, 'orange', 'black')
