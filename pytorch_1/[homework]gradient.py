
from copy import copy, deepcopy

import numpy as np
import matplotlib.pyplot as plt



TASK = 9

if TASK == 6:
    """"
    TASK 6 
    """



    def grad_descent_v2(f, df, low=None, high=None, callback=None):

        def find_local_min(f, deriv, low_local, high_local, x0 = None, iters=5000, lr=0.05):
            # функция для нахождения минимума функции f на промежутке (low_local, high_local)
            if x0 is None:
                x0 = np.random.uniform(low_local, high_local)
            x = x0
            for i in range(iters):
                try:
                    if abs(deriv(x)) < 1e-4:
                        #print("Выход по сходимости")
                        break
                    if abs(deriv(x)) > 4e+25:
                        break

                    k = 1 if deriv(x) > 0 else -1
                    x = x - k * lr * 1 / (i + 1)**0.5
                    x = np.clip(x, low_local, high_local)
                    #-2.91701
                    try:
                        callback(x, f(x))
                    except OverflowError:
                        #print(x , f)
                        pass
                    except TypeError as e:
                        #print(e, x, f)
                        pass
                except OverflowError:
                    #print(x, f)
                    pass
            return x

        optimum = list()
        # Разбейте отрезок [low, high] на 3-6 равных частей
        parts = 5
        dip = np.linspace(low, high, parts + 1)
        for i in range(parts):
            for j in range(1):
                for lr in [0.1, 0.01, 0.05, 0.001, 0.005, 0.0001]:
                    optimum.append(find_local_min(f, df, dip[i], dip[i+1], x0= dip[i]+ (dip[i+1]-dip[i])/2, iters = 100))


        # Для каждой части запустите find_local_min несколько
        # (преподавательский код запускает 10) раз
        optimum_f = f(np.array(optimum))
        best_i = np.argmin(optimum_f)
        best_estimate =  optimum[best_i]# Найдите общий минимум по всем запускам. Возможно, вы захотите
        #print(list(zip(optimum, optimum_f)))
        # использовать np.argmin
        return best_estimate


    def plot_convergence_1d(func, x_steps, y_steps, ax, grid=None, title=""):
        """
        Функция отрисовки шагов градиентного спуска.
        Не меняйте её код без необходимости!
        :param func: функция, которая минимизируется градиентным спуском
        :param x_steps: np.array(float) — шаги алгоритма по оси Ox
        :param y_steps: np.array(float) — шаги алгоритма по оси Оу
        :param ax: холст для отрисовки графика
        :param grid: np.array(float) — точки отрисовки функции func
        :param title: str — заголовок графика
        """
        ax.set_title(title, fontsize=16, fontweight="bold")

        if grid is None:
            grid = np.linspace(np.min(x_steps), np.max(x_steps), 100)

        fgrid = [func(item) for item in grid]
        ax.plot(grid, fgrid)
        yrange = np.max(fgrid) - np.min(fgrid)

        arrow_kwargs = dict(linestyle="--", color="grey", alpha=0.4)
        for i, _ in enumerate(x_steps):
            if i + 1 < len(x_steps):
                ax.arrow(
                    x_steps[i], y_steps[i],
                    x_steps[i + 1] - x_steps[i],
                    y_steps[i + 1] - y_steps[i],
                    **arrow_kwargs
                )

        n = len(x_steps)
        color_list = [(i / n, 0, 0, 1 - i / n) for i in range(n)]
        ax.scatter(x_steps, y_steps, c=color_list)
        ax.scatter(x_steps[-1], y_steps[-1], c="red")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")


    class LoggingCallback:
        """
        Класс для логирования шагов градиентного спуска.
        Сохраняет точку (x, f(x)) на каждом шаге.
        Пример использования в коде: callback(x, f(x))
        """
        def __init__(self):
            self.x_steps = []
            self.y_steps = []

        def __call__(self, x, y):
            self.x_steps.append(x)
            self.y_steps.append(y)


    def test_convergence_1d(grad_descent, test_cases, tol=1e-2, axes=None, grid=None):
        """
        Функция для проверки корректности вашего решения в одномерном случае.
        Она же используется в тестах на Stepik, так что не меняйте её код!
        :param grad_descent: ваша реализация градиентного спуска
        :param test_cases: dict(dict), тесты в формате dict с такими ключами:
            - "func" — функция (обязательно)
            - "deriv" — её производная (обязательно)
            - "start" — начальная точка start (м.б. None) (опционально)
            - "low", "high" — диапазон для выбора начальной точки (опционально)
            - "answer" — ответ (обязательно)
        При желании вы можете придумать и свои тесты.
        :param tol: предельное допустимое отклонение найденного ответа от истинного
        :param axes: матрица холстов для отрисовки, по ячейке на тест
        :param grid: np.array(float), точки на оси Ох для отрисовки тестов
        :return: флаг, корректно ли пройдены тесты, и дебажный вывод в случае неудачи
        """
        right_flag = True
        debug_log = []
        for i, key in enumerate(test_cases.keys()):
            # Формируем входные данные и ответ для алгоритма.
            answer = test_cases[key]["answer"]
            test_input = deepcopy(test_cases[key])
            del test_input["answer"]
            # Запускаем сам алгоритм.
            callback = LoggingCallback()  # Не забываем про логирование
            res_point = grad_descent(*test_input.values(), callback=callback)
            # Отрисовываем результаты.
            if axes is not None:
                ax = axes[np.unravel_index(i, shape=axes.shape)]
                x_steps = np.array(callback.x_steps)
                y_steps = np.array(callback.y_steps)
                if len(x_steps) > 0 :
                    plot_convergence_1d(
                        test_input["func"], x_steps, y_steps,
                        ax, grid, key
                    )
                ax.axvline(answer, 0, linestyle="--", c="red",
                            label=f"true answer = {answer}")
                ax.axvline(res_point, 0, linestyle="--", c="xkcd:tangerine",
                            label=f"estimate = {np.round(res_point, 3)}")
                ax.legend(fontsize=16)
            # Проверяем, что найдення точка достаточно близко к истинной
            if abs(answer - res_point) > tol or np.isnan(res_point):
                debug_log.append(
                    f"Тест '{key}':\n"
                    f"\t- ответ: {answer}\n"
                    f"\t- вывод алгоритма: {res_point}"
                )
                right_flag = False
        return right_flag, debug_log

    test_cases = {
        "poly1" : {
            "func" : lambda x: x**4 + 3 * x**3 + x**2 - 1.5 * x + 1,
            "deriv" : lambda x: 4 * x**3 + 9 * x**2 + 2 * x - 1.5,
            "low" : -4, "high" : 2, "answer" : -1.88
        },
        "poly2" : {
            "func" : lambda x: x**4 + 3 * x**3 + x**2 - 2 * x + 1.0,
            "deriv" : lambda x: 4 * x**3 + 9 * x**2 + 2 * x - 2.0,
            "low" : -3, "high" : 3, "answer" : 0.352
        },
        "another yet poly" : {
            "func" : lambda x: x**6 + x**4 - 10 * x**2 - x ,
            "deriv" : lambda x: 6 * x**5 + 4 * x**3 - 20 * x - 1,
            "low" : -2, "high" : 2, "answer" : 1.24829
        },
        "and another yet poly" : {
            "func" : lambda x: x**20 + x**2 - 20 * x + 10,
            "deriv" : lambda x: 20 * x**19 + 2 * x - 20,
            "low" : -0, "high" : 2, "answer" : 0.994502
        },
        "|x|/x^2 - x + sqrt(-x) + (even polynom)" : {
            "func" : lambda x: 5 * np.abs(x)/x**2 - 0.5 * x + 0.1 * np.sqrt(-x) + 0.01 * x**2 ,
            "deriv" : lambda x: -0.5 - 0.05/np.sqrt(-x) + 0.02 * x + 5/(x * np.abs(x)) - (10 * np.abs(x))/x**3,
            "low" : -4, "high" : -2, "answer": -2.91701
        },
    }

    tol = 1e-2 # желаемая точность

    fig, axes = plt.subplots(2, 3, figsize=(24, 8))
    fig.suptitle("Градиентный спуск, версия 2", fontweight="bold", fontsize=20)
    grid = np.linspace(-3, 3, 100)

    is_correct, debug_log = test_convergence_1d(
        grad_descent_v2, test_cases, tol,
        axes, grid
    )

    if not is_correct:
        print("Не сошлось. Дебажный вывод:")
        for log_entry in debug_log:
            print(log_entry)
    plt.show()

if TASK == 9:
    from sympy import *


    def numerical_derivative_2d(func, epsilon):
        """
        Функция для приближённого вычисления градиента функции двух переменных.
        :param func: np.array[2] -> float — произвольная дифференцируемая функция
        :param epsilon: float — максимальная величина приращения по осям
        :return: другая функция, которая приближённо вычисляет градиент в точке
        """

        def grad_func(x):
            """
            :param x: np.array[2] — точка, в которой нужно вычислить градиент
            :return: np.array[2] — приближённое значение градиента в этой точке
            """
            # < YOUR
            # CODE >

            return #< YOUR  CODE >

        return grad_func


    def grad_descent_2d(func, low, high, start=None, callback=None):
        """
        Реализация градиентного спуска для функций двух переменных

        Обратите внимание, что здесь градиент функции не дан.
        Его нужно вычислять приближённо.

        :param func: np.ndarray -> float — функция
        :param low: левая граница интервала по каждой из осей
        :param high: правая граница интервала по каждой из осей
        """
        eps = 1e-10
        df = numerical_derivative_2d(func, eps)

        if x0 is None:
            # Если точка не дана, сгенерируем случайную
            # из стандартного нормального распределения.
            # При таком подходе начальная точка может быть
            # любой, а не только из какого-то ограниченного диапазона
            # np.random.seed(179)
            x0 = np.random.uniform()

        x = x0

        callback(x, f(x))  # не забывайте логировать

        for i in range(iters):

            if abs(deriv(x)) < 1e-4:
                # print("Выход по сходимости")
                break

            x = x - lr * deriv(x)
            callback(x, f(x))

        return x

        return < YOUR CODE >

    def plot_convergence_2d(func, steps, ax, xlim, ylim, cmap="viridis", title=""):
        """
        Функция отрисовки шагов градиентного спуска.
        Не меняйте её код без необходимости!
        :param func: функция, которая минимизируется градиентным спуском
        :param steps: np.array[N x 2] — шаги алгоритма
        :param ax: холст для отрисовки графика
        :param xlim: tuple(float), 2 — диапазон по первой оси
        :param ylim: tuple(float), 2 — диапазон по второй оси
        :param cmap: str — название палитры
        :param title: str — заголовок графика
        """

        ax.set_title(title, fontsize=20, fontweight="bold")
        # Отрисовка значений функции на фоне
        xrange = np.linspace(*xlim, 100)
        yrange = np.linspace(*ylim, 100)
        grid = np.meshgrid(xrange, yrange)
        X, Y = grid
        fvalues = func(
            np.dstack(grid).reshape(-1, 2)
        ).reshape((xrange.size, yrange.size))
        ax.pcolormesh(xrange, yrange, fvalues, cmap=cmap, alpha=0.8)
        CS = ax.contour(xrange, yrange, fvalues)
        ax.clabel(CS, CS.levels, inline=True)
        # Отрисовка шагов алгоритма в виде стрелочек
        arrow_kwargs = dict(linestyle="--", color="black", alpha=0.8)
        for i, _ in enumerate(steps):
            if i + 1 < len(steps):
                ax.arrow(
                    *steps[i],
                    *(steps[i + 1] - steps[i]),
                    **arrow_kwargs
                )
        # Отрисовка шагов алгоритма в виде точек
        n = len(steps)
        color_list = [(i / n, 0, 0, 1 - i / n) for i in range(n)]
        ax.scatter(steps[:, 0], steps[:, 1], c=color_list, zorder=10)
        ax.scatter(steps[-1, 0], steps[-1, 1],
                   color="red", label=f"estimate = {np.round(steps[-1], 2)}")
        # Финальное оформление графиков
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_ylabel("$y$")
        ax.set_xlabel("$x$")
        ax.legend(fontsize=16)