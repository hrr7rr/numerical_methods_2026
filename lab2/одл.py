import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

# ==========================================
# ЧАСТИНА 1: Модель CPU = f(RPS) за таблицею
# ==========================================

# Імітація зчитування з CSV-файлу (ви можете замінити на pd.read_csv('ваш_файл.csv'))
csv_data = """RPS,CPU
50,20
100,35
200,60
400,110
800,210"""

print("--- 1. Вхідні дані ---")
df = pd.read_csv(io.StringIO(csv_data))
x_data = df['RPS'].values
y_data = df['CPU'].values
print(df.to_string(index=False))
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


# 2. Будуємо таблицю розділених різниць
F = divided_diff_table(x_data, y_data)

print("--- 2. Таблиця розділених різниць ---")
columns = ['y'] + [f'Різниця {j}-го пор.' for j in range(1, len(x_data))]
df_diff = pd.DataFrame(F, columns=columns)
df_diff.insert(0, 'x (RPS)', x_data)
# Виводимо, приховуючи нулі для краси (як у класичній таблиці)
print(df_diff.replace(0.0, '').to_string(index=False))
print("\n")

# 3. Обчислення прогнозу для 600 RPS
target_x = 600
predicted_cpu = newton_poly(target_x, x_data, F)
print("--- 3. Прогноз ---")
print(f"Розрахункове навантаження CPU при {target_x} RPS: {predicted_cpu:.2f}%\n")

# 4. Побудова графіка CPU(RPS)
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


# ==========================================
# ЧАСТИНА 2: Дослідження моделі CPU(RPS)
# 5, 10, 20 вузлів — на одному графіку
# ==========================================

nodes_list = [5, 10, 20]

a, b = min(x_data), max(x_data)

# густа сітка
x_dense = np.linspace(a, b, 1000)

# "Базова" модель (по початкових 5 експериментальних точках)
y_base = np.array([newton_poly(xi, x_data, F) for xi in x_dense])


# ЧАСТИНА 1: ГРАФІК ЯК НА ФОТО (Окреме вікно)
# =========================================================================
n_photo = 5
a, b = min(x_data), max(x_data)

# 1. Підготовка даних для графіка як на фото
x_nodes_photo = np.linspace(a, b, n_photo)
y_nodes_photo = np.array([newton_poly(xi, x_data, F) for xi in x_nodes_photo])
F_photo = divided_diff_table(x_nodes_photo, y_nodes_photo)

x_plot_fine = np.linspace(a, b, 1000)
y_plot_fine = np.array([newton_poly(xi, x_nodes_photo, F_photo) for xi in x_plot_fine])

# 2. Побудова
plt.figure(figsize=(7, 6))  # Розмір близький до квадрату, як на скріншоті
plt.plot(x_plot_fine, y_plot_fine, color='blue', linewidth=1.5)
plt.scatter(x_nodes_photo, y_nodes_photo, color='red', s=40, zorder=5)

# 3. Стилізація (максимально схожа на фото)
plt.title(f'{n_photo} вузлів (Степінь {n_photo - 1})', fontsize=14)
plt.grid(True, which='both', linestyle='-', color='gray', alpha=0.7)
plt.xlim(30, 850)
# На фото вісь Y починається приблизно від 10 до 220
plt.ylim(10, 220)

# Виводимо цей графік окремо
plt.show()

