import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

# Шаг 1: Загружаем данные из библиотеки Wine
wine = load_wine()
data = pd.DataFrame(wine.data, columns=wine.feature_names)
data['target'] = wine.target
data['target_name'] = data['target'].apply(lambda x: wine.target_names[x])

# Шаг 2: Строим графики на базе датасета Wine

# График 1. Гистограмма  (Alcohol Content Distribution):
# создаем размер графика, выбираем данные,  цвет и прозрачность, количество столбов
# задаем название графика, осей, и легенду
plt.figure(figsize=(10, 6))
for target_name, group in data.groupby('target_name'):
    plt.hist(group['alcohol'], bins=15, alpha=0.7, label=target_name)
plt.title("Alcohol Content Distribution by Wine Type", fontsize=16)
plt.xlabel("Alcohol Content (%)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend(title="Wine Type")
# создаем сетку и выводим график
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# График 2. Диаграмма рассеяния (Flavanoids vs Color Intensity):
# создаем размер графика, задаем цвета для групп, прозрачностью выбираем нужные данные
plt.figure(figsize=(10, 6))
colors = ['red', 'green', 'blue']
for i, target_name in enumerate(wine.target_names):
    subset = data[data['target'] == i]
    plt.scatter(subset['flavanoids'], subset['color_intensity'],
                label=target_name, color=colors[i], alpha=0.7)
# задаем название графика и осей, легенду
plt.title("Flavanoids vs Color Intensity", fontsize=16)
plt.xlabel("Flavanoids", fontsize=12)
plt.ylabel("Color Intensity", fontsize=12)
plt.legend(title="Wine Type")
# создаем сетку и выводим график, автоматическая настройка промежутка
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# График 3. Boxplot - ящик с усами: (Malic Acid by Wine Type)
# выбираем данные, задаем размер графика, форма ящика, цвет заливки

plt.figure(figsize=(10, 6))
data.boxplot(column='malic_acid', by='target_name', grid=False, patch_artist=True,
             boxprops=dict(facecolor='lightblue'))
# задаем название графика и осей, легенду
plt.title("Boxplot of Malic Acid Content by Wine Type", fontsize=16)
plt.suptitle("")  # Убираем лишний заголовок
plt.xlabel("Wine Type", fontsize=12)
plt.ylabel("Malic Acid Content", fontsize=12)
# автоматическая настройка промежутка
plt.tight_layout()
# выводим график
plt.show()

# График 4. Гистограмма (Total Phenols Distribution):
# создаем размер графика, выбираем данные,  цвет и прозрачность, количество столбов
# задаем название графика, осей, и легенду

plt.figure(figsize=(10, 6))
plt.hist(data['total_phenols'], bins=20, color='orange', edgecolor='black', alpha=0.7)
plt.title("Distribution of Total Phenols", fontsize=16)
plt.xlabel("Total Phenols", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
# создаем сетку, выводим график, автоматическая настройка промежутка
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
