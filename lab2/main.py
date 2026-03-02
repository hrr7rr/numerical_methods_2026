import numpy as np
import matplotlib.pyplot as plt
import math
import csv

# Функція для зчитування даних з CSV файлу
def read_data(filename):
    x_val = []
    y_val = []
    with open(filename, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Звертання до колонок 'RPS' та 'CPU'
            x_val.append(float(row['RPS']))
            y_val.append(float(row['CPU']))
    return x_val, y_val


# Зчитування даних з файлу
x, y = read_data("data.csv")

x_data = np.array(x)
y_data = np.array(y)

print(" Вхідні дані")
print(f"{'RPS':<8} {'CPU'}")
for xi, yi in zip(x_data, y_data):
    print(f"{int(xi):<8} {int(yi)}")
print("\n")


# Функція для побудови таблиці розділених різниць
def divided_diff_table(x, y):
    n = len(x)
    F = np.zeros((n, n))
    F[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            F[i, j] = (F[i + 1, j - 1] - F[i, j - 1]) / (x[i + j] - x[i])
    return F


# Обчислення функції w_k(x) = (x-x0)*(x-x1)*...*(x-x_{k-1})
def omega(x, x_nodes, k):
    result = 1.0
    for i in range(k):
        result *= (x - x_nodes[i])
    return result


# Знаходження значення інтерполяційного многочлена Ньютона N_n(x)
def newton_poly(x, x_nodes, F_table):
    n = len(x_nodes)
    result = F_table[0, 0]
    for k in range(1, n):
        result += F_table[0, k] * omega(x, x_nodes, k)
    return result


# Будуємо таблицю розділених різниць
F = divided_diff_table(x_data, y_data)

print("Таблиця розділених різниць")
n_points = len(x_data)
columns = ['x (RPS)', 'y'] + [f'Різн. {j}-го пор.' for j in range(1, n_points)]
print("".join([f"{col:<18}" for col in columns]))

# Виведення рядків
for i in range(n_points):
    row_str = f"{x_data[i]:<18.1f}"
    for j in range(n_points):
        if i + j < n_points:
            row_str += f"{F[i, j]:<18.5f}"
        else:
            row_str += f"{'':<18}"
    print(row_str)
print("\n")

# Обчислення прогнозу для 600 RPS
target_x = 600

# Функція для методу факторіальних многочленів
def factorial_poly_newton(x_target, x_nodes, y_nodes):
    t_nodes = np.log2(x_nodes / 50)
    t_target = np.log2(x_target / 50)
    n = len(y_nodes)

    delta = np.zeros((n, n))
    delta[:, 0] = y_nodes
    for j in range(1, n):
        for i in range(n - j):
            delta[i, j] = delta[i + 1, j - 1] - delta[i, j - 1]

    q = t_target
    result = delta[0, 0]

    for k in range(1, n):
        fact_poly = 1.0
        for i in range(k):
            fact_poly *= (q - i)
        result += (delta[0, k] * fact_poly) / math.factorial(k)

    return result


# Обчислення прогнозів двома методами
predicted_cpu_newton = newton_poly(target_x, x_data, F)
predicted_cpu_factorial = factorial_poly_newton(target_x, x_data, y_data)

print(" Прогноз ")
print(f"1. Метод Ньютона (розділені різниці): {predicted_cpu_newton:.2f}%")
print(f"2. Метод факторіальних многочленів: {predicted_cpu_factorial:.2f}%\n")

# Зберігаємо результат Ньютона у змінну predicted_cpu для графіка
predicted_cpu = predicted_cpu_newton

# Побудова графіка CPU(RPS)
plt.figure(figsize=(10, 6))
x_plot = np.linspace(min(x_data), max(x_data), 500)
y_plot = [newton_poly(xi, x_data, F) for xi in x_plot]

plt.plot(x_plot, y_plot, 'b-', label='Інтерполяційний многочлен Ньютона $N_n(x)$')
plt.scatter(x_data, y_data, color='red', s=50, zorder=5, label='Вихідні дані (вузли)')
plt.scatter([target_x], [predicted_cpu], color='green', marker='*', s=200, zorder=6,
            label=f'Прогноз (RPS={target_x}, CPU={predicted_cpu:.1f}%)')

plt.title('Графік залежності CPU від RPS', fontsize=14)
plt.xlabel('RPS', fontsize=12)
plt.ylabel('CPU (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=11)
plt.show()

# 5, 10, 20 вузлів — графіки

nodes_list = [5, 10, 20]
a, b = min(x_data), max(x_data)

x_dense = np.linspace(a, b, 1000)

# Базова модель (по реальних 5 точках)
y_base = np.array([newton_poly(xi, x_data, F) for xi in x_dense])

plt.figure(figsize=(18, 5))

errors_dict = {}

print(" Дослідження похибки (5, 10, 20 вузлів)")
print("Оскільки базова функція є поліномом 4-го степеня, похибка є суто обчислювальною (машинною).")

for idx, n in enumerate(nodes_list):
    plt.subplot(1, 3, idx + 1)

    # рівномірні вузли
    x_n = np.linspace(a, b, n)
    y_n = np.array([newton_poly(xi, x_data, F) for xi in x_n])

    # новий поліном
    F_n = divided_diff_table(x_n, y_n)
    y_interp = np.array([newton_poly(xi, x_n, F_n) for xi in x_dense])

    error = np.abs(y_base - y_interp)
    max_error = np.max(error)
    errors_dict[n] = error

    # Виведення похибок у консоль
    print(f"Кількість вузлів: {n:<2} | Макс. похибка: {max_error:.2e}")

    # графік
    plt.plot(x_dense, y_interp, 'b-')
    plt.scatter(x_n, y_n, color='red', zorder=5)

    plt.title(f'{n} вузлів (Степінь {n - 1})\nMax похибка: {max_error:.2e}')
    plt.xlabel('RPS')
    plt.ylabel('CPU (%)')
    plt.grid(True, alpha=0.6)

plt.tight_layout()
plt.show()

# графік похибок

plt.figure(figsize=(10, 6))

for n in nodes_list:
    plt.plot(x_dense, errors_dict[n], label=f'n = {n}')

plt.title('Похибка інтерполяції CPU(RPS)')
plt.xlabel('RPS')
plt.ylabel('|N_base(x) - N_n(x)|')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()