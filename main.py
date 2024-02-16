import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def load_and_describe_data(file_path):
    """Загрузка данных и их описание."""
    # Загрузка данных
    data_frame = pd.read_csv(file_path)

    # Вывод первых пяти строк DataFrame
    print("Первые пять строк DataFrame:")
    print(data_frame.head())

    # Вывод статистических данных DataFrame
    print("\nСтатистические данные DataFrame:")
    print(data_frame.describe())

    return data_frame


def analyze_gender_data(data_frame):
    """Анализ данных по полу."""
    # Создание DataFrame для мужчин и женщин
    men_df = data_frame[data_frame.Gender == 'Male']
    women_df = data_frame[data_frame.Gender == 'Female']

    # Анализ данных и создание графиков
    for gender_df, gender_name in zip([men_df, women_df], ['Male', 'Female']):
        mean_smarts = gender_df[["PIQ", "FSIQ", "VIQ"]].mean(axis=1)
        plt.scatter(mean_smarts, gender_df["MRI_Count"])
        plt.title(f'Корреляция между средним интеллектом и MRI Count для {gender_name}')
        plt.xlabel('Средний интеллект')
        plt.ylabel('MRI Count')
        plt.savefig(f'{gender_name.lower()}_correlation.png')
        plt.close()


def compute_and_display_correlation(data_frame, gender=None):
    """Вычисление и отображение корреляции."""
    # Исключение нечисловых данных
    numeric_df = data_frame.select_dtypes(include=[np.number])

    if gender:
        print(f"\nТаблица корреляции для {gender}:")
    else:
        print("\nТаблица корреляции для всего DataFrame:")
    print(numeric_df.corr(method='pearson'))

    # Отображение тепловых карт корреляции
    corr = numeric_df.corr()
    sns.heatmap(corr)
    plt.title(f'Тепловая карта корреляции для {gender if gender else "всех"}')
    plt.savefig(f'heatmap_{gender.lower() if gender else "all"}.png')
    plt.close()


# Основной блок
if __name__ == "__main__":
    brain_file = 'brainsize.txt'
    brain_frame = load_and_describe_data(brain_file)

    analyze_gender_data(brain_frame)

    compute_and_display_correlation(brain_frame)
    compute_and_display_correlation(brain_frame[brain_frame.Gender == 'Female'], 'Female')
    compute_and_display_correlation(brain_frame[brain_frame.Gender == 'Male'], 'Male')
