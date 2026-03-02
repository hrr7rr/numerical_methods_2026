import numpy as np
import math

# ==========================================
# Вхідні дані
# ==========================================
x_data = np.array([50.0, 100.0, 200.0, 400.0, 800.0])
y_data = np.array([20.0, 35.0, 60.0, 110.0, 210.0])
target_x = 600.0

print(f"--- Прогноз CPU для {target_x} RPS ---\n")


# ==========================================
# МЕТОД 1: Розділені різниці (Многочлен Ньютона)
# Працює безпосередньо з x_data
# ==========================================
def divided_diff_table(x, y):
    n = len(x)
    F = np.zeros((n, n))
    F[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            F[i, j] = (F[i + 1, j - 1] - F[i, j - 1]) / (x[i + j] - x[i])
    return F


def newton_poly_divided(x_target, x_nodes, F_table):
    n = len(x_nodes)
    result = F_table[0, 0]
    for k in range(1, n):
        # Узагальнений многочлен (x-x0)(x-x1)...
        omega = 1.0
        for i in range(k):
            omega *= (x_target - x_nodes[i])
        result += F_table[0, k] * omega
    return result


F_div = divided_diff_table(x_data, y_data)
cpu_method_1 = newton_poly_divided(target_x, x_data, F_div)
print(f"1. Метод Ньютона (розділені різниці): {cpu_method_1:.2f}%")


# ==========================================
# МЕТОД 2: Факторіальні многочлени (Скінченні різниці)
# Вимагає рівномірної сітки. Використовуємо t = log2(x / 50)
# ==========================================
def factorial_poly_newton(x_target, x_nodes, y_nodes):
    # Переходимо до рівномірної шкали t (крок h = 1, t0 = 0)
    t_nodes = np.log2(x_nodes / 50)
    t_target = np.log2(x_target / 50)
    n = len(y_nodes)

    # Будуємо таблицю скінченних різниць (Δy)
    delta = np.zeros((n, n))
    delta[:, 0] = y_nodes
    for j in range(1, n):
        for i in range(n - j):
            delta[i, j] = delta[i + 1, j - 1] - delta[i, j - 1]

    # Перша інтерполяційна формула Ньютона
    # q = (t_target - t0) / h. Оскільки h=1 і t0=0, то q = t_target
    q = t_target
    result = delta[0, 0]

    for k in range(1, n):
        # Факторіальний многочлен: q^(k) = q * (q-1) * ... * (q-k+1)
        fact_poly = 1.0
        for i in range(k):
            fact_poly *= (q - i)

        # Додаємо член: (Δ^k y0 * q^(k)) / k!
        result += (delta[0, k] * fact_poly) / math.factorial(k)

    return result


cpu_method_2 = factorial_poly_newton(target_x, x_data, y_data)
print(f"2. Метод факторіальних многочленів: {cpu_method_2:.2f}%")