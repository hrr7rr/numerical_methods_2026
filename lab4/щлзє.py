
import numpy as np
import matplotlib.pyplot as plt
import math

# Функція вологості
def M(t):
    return 50 * math.exp(-0.1 * t) + 5 * math.sin(t)

# Аналітична похідна
def M_derivative(t):
    return -5 * math.exp(-0.1 * t) + 5 * math.cos(t)

# Точка
t0 = 1.0

# Обчислення
exact_value = M_derivative(t0)

print("Явний вираз похідної:")
print("M'(t) = -5e^(-0.1t) + 5cos(t)")

print("\nТочне значення в точці t0 = 1:")
print("M'(1) =", exact_value)

import numpy as np
import math
import matplotlib.pyplot as plt

# ==========================================
# Функція та точна похідна
# ==========================================

def f(t):
    return 50 * math.exp(-0.1 * t) + 5 * math.sin(t)

def f_exact_derivative(t):
    return -5 * math.exp(-0.1 * t) + 5 * math.cos(t)

x0 = 1.0
exact = f_exact_derivative(x0)


# --- 1. Задаємо функцію та її аналітичну похідну ---
def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

def M_derivative(t):
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

t0 = 1.0
exact_value = M_derivative(t0)
print(f"Точне значення похідної M'({t0}) = {exact_value:.6f}")

# --- 2. Дослідження похибки від кроку h ---
h_values = np.logspace(-20, 3, 100)  # кроки від 1e-20 до 1e3
errors = []

for h in h_values:
    # центральна різниця
    y_prime = (M(t0 + h) - M(t0 - h)) / (2*h)
    errors.append(abs(y_prime - exact_value))

# Вибір оптимального h (мінімальна похибка)
min_error_index = np.argmin(errors)
h_opt = h_values[min_error_index]
print(f"Оптимальний крок h_opt = {h_opt:.6e}, похибка = {errors[min_error_index]:.6e}")

# --- 3. Фіксований крок h = 1e-3 ---
h = 1e-3
y_h = (M(t0 + h) - M(t0 - h)) / (2*h)
y_2h = (M(t0 + 2*h) - M(t0 - 2*h)) / (4*h)
R1 = abs(y_h - exact_value)

print(f"\nЧисельне диференціювання:")
print(f"y'(h={h}) = {y_h:.6f}, похибка R1 = {R1:.6e}")

# --- 4. Метод Рунге-Ромберга ---
y_R = y_h + (y_h - y_2h) / 3  # для центральної різниці p=2
R2 = abs(y_R - exact_value)
print(f"\nМетод Рунге-Ромберга:")
print(f"y'_RR = {y_R:.6f}, похибка R2 = {R2:.6e}")

# --- 5. Метод Ейткена (h, 2h, 4h) ---
y_4h = (M(t0 + 4*h) - M(t0 - 4*h)) / (8*h)
y_E = ((y_2h**2) - y_4h*y_h) / (2*y_2h - (y_4h + y_h))
# оцінка порядку точності p
p = np.log(abs((y_4h - y_2h) / (y_2h - y_h))) / np.log(2)
R3 = abs(y_E - exact_value)

print(f"\nМетод Ейткена:")
print(f"y'_Aitken = {y_E:.6f}, похибка R3 = {R3:.6e}, порядок точності p ≈ {p:.3f}")

# --- 6. Графік похибки від кроку ---
plt.figure(figsize=(8,5))
plt.loglog(h_values, errors, label="Похибка |y' - y_exact|")
plt.scatter(h_opt, errors[min_error_index], color='red', label=f"h_opt={h_opt:.1e}")
plt.xlabel("Крок h")
plt.ylabel("Похибка")
plt.title("Залежність похибки від кроку h")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()

# --- 7. Висновки: оптимальні режими поливу ---
if exact_value > 0:
    print("\nГрунт швидко втрачає вологу — поливати частіше.")
else:
    print("\nГрунт вологіший — поливати рідше.")