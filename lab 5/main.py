import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


#  Задана функція навантаження на сервер

def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12) ** 2)


# Інтервал інтегрування
a, b = 0.0, 24.0

#  Точне значення інтегралу
exact_value, exact_err = quad(f, a, b)


# Складова формула Сімпсона

def simpson_composite(f, a, b, n):
    """Обчислення інтегралу складовою формулою Сімпсона для n відрізків (n - парне)."""
    if n % 2 != 0:
        n += 1  # n має бути парним
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    # Застосування формули Сімпсона
    integral = y[0] + y[-1]
    integral += 4 * np.sum(y[1:-1:2])  # Непарні індекси
    integral += 2 * np.sum(y[2:-2:2])  # Парні індекси

    return integral * h / 3


# Дослідження залежності точності від n
target_eps = 1e-6
n_values = np.arange(2, 102, 2)
errors = []
target_n = None
target_integral = None

for n in n_values:
    val = simpson_composite(f, a, b, n)
    err = abs(exact_value - val)
    errors.append(err)
    if target_n is None and err <= target_eps:
        target_n = n
        target_integral = val

#  Обчислення похибки при конкретному n

n_base = 10
I_n = simpson_composite(f, a, b, n_base)
err_n = abs(exact_value - I_n)

#  Метод Рунге-Ромберга

I_2n = simpson_composite(f, a, b, n_base * 2)
# Порядок точності методу Сімпсона p = 4
runge_romberg_val = I_2n + (I_2n - I_n) / (2 ** 4 - 1)
runge_romberg_err = abs(exact_value - runge_romberg_val)

#  Метод Ейткена
I_4n = simpson_composite(f, a, b, n_base * 4)

# Обчислення порядку методу p
# m = (I(2n) - I(n)) / (I(4n) - I(2n))
diff1 = I_2n - I_n
diff2 = I_4n - I_2n
if diff2 != 0:
    m = diff1 / diff2
    p_aitken = np.log(abs(m)) / np.log(2)
else:
    p_aitken = float('inf')

# Уточнене значення
if diff1 != diff2:
    aitken_val = I_4n + (diff2 ** 2) / (diff1 - diff2)
else:
    aitken_val = I_4n

aitken_err = abs(exact_value - aitken_val)


#  Адаптивний алгоритм
def adaptive_simpson(f, a, b, eps):
    """
    Адаптивний алгоритм інтегрування з підрахунком кількості викликів функції.
    """
    calls = 0

    def recursive_step(a, b, fa, fb, fc, S, eps):
        nonlocal calls
        c = (a + b) / 2
        h = b - a
        d = (a + c) / 2
        e = (c + b) / 2

        fd, fe = f(d), f(e)
        calls += 2

        S_left = (h / 12) * (fa + 4 * fd + fc)
        S_right = (h / 12) * (fc + 4 * fe + fb)
        S2 = S_left + S_right

        # Умова збіжності за правилом Рунге
        if abs(S2 - S) <= 15 * eps:
            return S2 + (S2 - S) / 15
        else:
            return recursive_step(a, c, fa, fc, fd, S_left, eps / 2) + \
                recursive_step(c, b, fc, fb, fe, S_right, eps / 2)

    # Ініціалізація
    c = (a + b) / 2
    fa, fb, fc = f(a), f(b), f(c)
    calls += 3
    S_init = ((b - a) / 6) * (fa + 4 * fc + fb)

    result = recursive_step(a, b, fa, fb, fc, S_init, eps)
    return result, calls


adapt_eps = 1e-6
adaptive_val, adaptive_calls = adaptive_simpson(f, a, b, adapt_eps)
adaptive_err = abs(exact_value - adaptive_val)

# Виведення
print(f" Точне значення інтегралу (еталон): {exact_value:.10f}")
print(f"Цільова точність: {target_eps}")
print(f"   Досягається при кількості розбиттів n = {target_n}")
print(f"   Значення інтегралу при цьому: {target_integral:.10f}")
print(f" Метод Сімпсона (n = {n_base}):")
print(f"   Значення: {I_n:.10f}, Похибка: {err_n:.2e}")
print(f" Метод Рунге-Ромберга (використано n={n_base} та 2n={n_base * 2}):")
print(f"   Значення: {runge_romberg_val:.10f}, Похибка: {runge_romberg_err:.2e}")
print(f" Метод Ейткена (використано n={n_base}, 2n={n_base * 2}, 4n={n_base * 4}):")
print(f"   Експериментальний порядок методу p: {p_aitken:.4f}")
print(f"   Значення: {aitken_val:.10f}, Похибка: {aitken_err:.2e}")
print(f" Адаптивний алгоритм (задана точність eps = {adapt_eps}):")
print(f"   Значення: {adaptive_val:.10f}, Похибка: {adaptive_err:.2e}")
print(f"   Кількість обчислень підінтегральної функції: {adaptive_calls}")


#  Побудова графіків
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Графік 1: Функція
x_plot = np.linspace(a, b, 1000)
y_plot = f(x_plot)
ax1.plot(x_plot, y_plot, 'b', label=r'$f(x)$')
ax1.fill_between(x_plot, y_plot, alpha=0.2, color='blue')
ax1.set_title('Графік функції навантаження на сервер')
ax1.set_xlabel('Час, x (год)')
ax1.set_ylabel('Навантаження, f(x)')
ax1.grid(True)
ax1.legend()

# Графік 2: Залежність похибки від n
ax2.plot(n_values, errors, 'r.-', label='Похибка')
if target_n:
    ax2.axvline(x=target_n, color='g', linestyle='--', label=f'Досягнення eps (n={target_n})')
ax2.set_title('Залежність похибки від кількості розбиттів n')
ax2.set_xlabel('Кількість розбиттів, n')
ax2.set_ylabel('Абсолютна похибка (логарифмічна шкала)')
ax2.set_yscale('log')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

# Дослідження адаптивного алгоритму для різних eps
eps_values = np.logspace(-2, -8, 10)  # від 1e-2 до 1e-8
errors_adapt = []
calls = []

for eps in eps_values:
    val, c = adaptive_simpson(f, a, b, eps)
    err = abs(exact_value - val)

    errors_adapt.append(err)
    calls.append(c)

