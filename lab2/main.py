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

# ============================================
# Дослідження впливу кроку
# Фіксований інтервал [a, b], різна кількість вузлів
# ============================================

print("\n Дослідження впливу кроку (фіксований інтервал)")

interval_a = min(x_data)
interval_b = max(x_data)

nodes_variants = [5, 8, 12, 16, 20, 25, 30]
step_results = []

plt.figure(figsize=(10, 6))

for n in nodes_variants:
    # Крок
    h = (interval_b - interval_a) / (n - 1)

    # Рівномірні вузли
    x_nodes_step = np.linspace(interval_a, interval_b, n)
    y_nodes_step = np.array([newton_poly(xi, x_data, F) for xi in x_nodes_step])

    # Побудова нового полінома
    F_step = divided_diff_table(x_nodes_step, y_nodes_step)
    y_interp_step = np.array([newton_poly(xi, x_nodes_step, F_step) for xi in x_dense])

    # Похибка відносно базової моделі
    error_step = np.abs(y_base - y_interp_step)
    max_error_step = np.max(error_step)

    step_results.append((n, h, max_error_step))

    print(f"n = {n:<3} | крок h = {h:<8.3f} | Макс. похибка = {max_error_step:.2e}")

# Побудова графіка залежності похибки від кроку
steps = [item[1] for item in step_results]
errors = [item[2] for item in step_results]

plt.plot(steps, errors, marker='o')
plt.title('Вплив кроку h на максимальну похибку')
plt.xlabel('Крок h')
plt.ylabel('Максимальна похибка')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# ============================================
# Дослідження впливу кількості вузлів
# Фіксований крок h, змінний інтервал
# ============================================

print("\n Дослідження: фіксований крок, змінний інтервал")

h_fixed = (b - a) / 20
print(f"Фіксований крок h = {h_fixed:.3f}")

interval_lengths = []
nodes_counts = []
max_errors = []

plt.figure(figsize=(10, 6))

for k in range(5, 21):  # різна кількість вузлів
    # новий інтервал [a, a + k*h]
    b_new = a + (k - 1) * h_fixed

    if b_new > b:
        break

    x_nodes_var = np.arange(a, b_new + h_fixed, h_fixed)
    y_nodes_var = np.array([newton_poly(xi, x_data, F) for xi in x_nodes_var])

    F_var = divided_diff_table(x_nodes_var, y_nodes_var)
    x_test = np.linspace(a, b_new, 1000)

    y_interp_var = np.array([newton_poly(xi, x_nodes_var, F_var) for xi in x_test])
    y_true_var = np.array([newton_poly(xi, x_data, F) for xi in x_test])

    error_var = np.abs(y_true_var - y_interp_var)
    max_error_var = np.max(error_var)

    interval_length = b_new - a
    n_nodes = len(x_nodes_var)

    interval_lengths.append(interval_length)
    nodes_counts.append(n_nodes)
    max_errors.append(max_error_var)

    print(f"Інтервал: [{a:.1f}, {b_new:.1f}] | "
          f"Довжина = {interval_length:<8.2f} | "
          f"Вузлів = {n_nodes:<3} | "
          f"Макс. похибка = {max_error_var:.2e}")

plt.plot(nodes_counts, max_errors, marker='o')
plt.title('Вплив кількості вузлів на похибку\n(фіксований крок)')
plt.xlabel('Кількість вузлів')
plt.ylabel('Максимальна похибка')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# ============================================
# Аналіз ефекту Рунге
# ============================================

print("\n Аналіз ефекту Рунге")

# Класична функція Рунге
def runge_function(x):
    return 1 / (1 + 25 * x**2)

# Інтервал [-1, 1]
a_runge = -1
b_runge = 1

x_dense_runge = np.linspace(a_runge, b_runge, 1000)
y_true_runge = runge_function(x_dense_runge)

nodes_runge_list = [5, 10, 15, 20]

plt.figure(figsize=(18, 5))

for idx, n in enumerate(nodes_runge_list):
    plt.subplot(1, 4, idx + 1)

    # Рівномірні вузли
    x_nodes_runge = np.linspace(a_runge, b_runge, n)
    y_nodes_runge = runge_function(x_nodes_runge)

    # Інтерполяція Ньютона
    F_runge = divided_diff_table(x_nodes_runge, y_nodes_runge)
    y_interp_runge = np.array([newton_poly(xi, x_nodes_runge, F_runge)
                               for xi in x_dense_runge])

    # Похибка
    error_runge = np.abs(y_true_runge - y_interp_runge)
    max_error_runge = np.max(error_runge)

    print(f"Вузлів: {n:<2} | Макс. похибка: {max_error_runge:.2e}")

    # Графік
    plt.plot(x_dense_runge, y_true_runge)
    plt.plot(x_dense_runge, y_interp_runge)
    plt.scatter(x_nodes_runge, y_nodes_runge, zorder=5)

    plt.title(f'n = {n}\nMax похибка: {max_error_runge:.2e}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)

plt.tight_layout()
plt.show()

# Графік похибки окремо

plt.figure(figsize=(10, 6))

for n in nodes_runge_list:
    x_nodes_runge = np.linspace(a_runge, b_runge, n)
    y_nodes_runge = runge_function(x_nodes_runge)
    F_runge = divided_diff_table(x_nodes_runge, y_nodes_runge)

    y_interp_runge = np.array([newton_poly(xi, x_nodes_runge, F_runge)
                               for xi in x_dense_runge])

    error_runge = np.abs(y_true_runge - y_interp_runge)
    plt.plot(x_dense_runge, error_runge, label=f'n = {n}')

plt.title('Похибка інтерполяції (ефект Рунге)')
plt.xlabel('x')
plt.ylabel('|f(x) - P_n(x)|')
plt.legend()
plt.grid(True)
plt.show()
# ============================================
# Порівняння з методом Лагранжа
# ============================================

print("\n Порівняння: Метод Ньютона vs Метод Лагранжа")

# Реалізація полінома Лагранжа
def lagrange_poly(x, x_nodes, y_nodes):
    n = len(x_nodes)
    result = 0.0
    for i in range(n):
        term = y_nodes[i]
        for j in range(n):
            if i != j:
                term *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += term
    return result


# Використаємо ті ж вузли, що й початкові
x_nodes = x_data
y_nodes = y_data

x_dense_compare = np.linspace(min(x_nodes), max(x_nodes), 1000)

# Обчислення значень
y_newton = np.array([newton_poly(xi, x_nodes, F) for xi in x_dense_compare])
y_lagrange = np.array([lagrange_poly(xi, x_nodes, y_nodes) for xi in x_dense_compare])

# Різниця між методами
difference = np.abs(y_newton - y_lagrange)
max_diff = np.max(difference)

print(f"Максимальна різниця між методами: {max_diff:.2e}")

# Графік порівняння
plt.figure(figsize=(10, 6))
plt.plot(x_dense_compare, y_newton, label='Ньютон')
plt.plot(x_dense_compare, y_lagrange, linestyle='--', label='Лагранж')
plt.scatter(x_nodes, y_nodes, zorder=5)

plt.title('Порівняння методів Ньютона та Лагранжа')
plt.xlabel('RPS')
plt.ylabel('CPU (%)')
plt.legend()
plt.grid(True)
plt.show()


# Графік різниці
plt.figure(figsize=(10, 6))
plt.plot(x_dense_compare, difference)
plt.title('Абсолютна різниця між поліномами')
plt.xlabel('RPS')
plt.ylabel('|N(x) - L(x)|')
plt.grid(True)
plt.show()