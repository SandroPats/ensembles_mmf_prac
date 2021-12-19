from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib


matplotlib.style.use('seaborn')


def plot_and_save_validation(val_score, name):
    fig = plt.figure(figsize=(8, 6))
    plt.title('Зависимость RMSE на валидационной выборке от числа деревьев')
    plt.plot(range(1, len(val_score) + 1),
             val_score)
    step = len(val_score) // 10
    plt.xticks(range(1, len(val_score) + 1, step))
    plt.xlabel('n_estimators')
    plt.ylabel('RMSE, $')
    plt.savefig(name)
