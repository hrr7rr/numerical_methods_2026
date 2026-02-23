import requests
import numpy as np
import matplotlib.pyplot as plt

# 1. Запит до Open-Elevation API


url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"

try: # Виконуємо GET-запит до API
    response = requests.get(url)
    data = response.json()
    results = data["results"]
except Exception as e:
    print(f"Помилка запиту: {e}")


# 2. Обчислення відстаней між точками горизонт шлях (Haversine)
#
# Функція для обчислення відстані між двома GPS точками
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Радіус Землі в метрах
    phi1, phi2 = np.radians(lat1), np.radians(lat2) # Переводимо широти у радіан
    dphi = np.radians(lat2 - lat1) # Різниця широт у радіанах
    dlambda = np.radians(lon2 - lon1)  # Різниця довгот у радіанах
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) # Відстань у метрах
# Створюємо списки координат і висот
coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = [p["elevation"] for p in results]

# Обчислюємо кумулятивну відстань
distances = [0]
for i in range(1, len(coords)):
    d = haversine(*coords[i - 1], *coords[i])
    distances.append(distances[-1] + d)

x = np.array(distances) # Відстань від старту (вісь X)
y = np.array(elevations) #Висота над рівнем моря (вісь Y)
n = len(x) # Кількість вузлів

# 3. Побудова кубічного сплайна (Метод прогонки)
#КУБІЧНИЙ СПЛАЙН (Щоб графік був плавний)
h = np.diff(x) # Відстані між сусідніми вузлами (h[i] = x[i+1] - x[i])
# Ініціалізація трьохдіагональної системи для знаходження M (других похідних)
A = np.zeros(n) # піддіагональ
B = np.zeros(n) # головна діагональ
C = np.zeros(n) # наддіагональ
D = np.zeros(n) # праві частини рівнянь

# Натуральний сплайн: друга похідна в кінцях дорівнює 0
 # Тому на кінцях B[0] = B[n-1] = 1, а D[0] = D[n-1] = 0 (за замовчуванням)
B[0] = 1
B[-1] = 1

#Формування внутрішніх рядків трьохдіагональної системи
 # Це рівняння походять із умови згладженості сплайна (похідні суміжних інтервалів)

for i in range(1, n - 1):
    A[i] = h[i - 1] # піддіагонал
    B[i] = 2 * (h[i - 1] + h[i])  # головна діагональ
    C[i] = h[i] # наддіагональ
    # Різниця нахилів сусідніх відрізків, помножена на 6. наскільки різко змінюється нахил
    D[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

# Прямий хід рішення трьохдіагональної системи методом прогонки (forward elimination)
for i in range(1, n):
    m = A[i] / B[i - 1] # коефіцієнт для виключення піддіагоналі
    B[i] = B[i] - m * C[i - 1] # модифікація головної діагоналі
    D[i] = D[i] - m * D[i - 1] # модифікація правої частини

# Зворотний хід (знаходження других похідних M)
M = np.zeros(n)    # масив других похідних
M[-1] = D[-1] / B[-1]  # останнє значення M

for i in range(n - 2, -1, -1): # рухаємося з кінця до початку

    M[i] = (D[i] - C[i] * M[i + 1]) / B[i]

# Обчислення коефіцієнтів a, b, c, d сплайна для формули: y = a + bx + cx^2 + dx^3
a_coeff = y[:-1] # коефіцієнт a = значення функції в лівому кінці інтервалу
c_coeff = M[:-1] / 2  # коефіцієнт c = половина другої похідної
d_coeff = np.diff(M) / (6 * h) # коефіцієнт d = кубічний коефіцієнт
b_coeff = (np.diff(y) / h) - (h * (2 * M[:-1] + M[1:]) / 6) # коефіцієнт b = лінійний член (нахил)


#
# Функції інтерполяції та аналізу

def spline_eval(xi):
    # Обчислення значення сплайна в точці xi знаходить висоту в будь-якому метрі дистанції
    if xi <= x[0]: return y[0]
    if xi >= x[-1]: return y[-1]
    idx = np.searchsorted(x, xi) - 1
    dx = xi - x[idx]
    return a_coeff[idx] + b_coeff[idx] * dx + c_coeff[idx] * dx ** 2 + d_coeff[idx] * dx ** 3


def spline_derivative(xi):
    # Знаходить нахил (градієнт) — наскільки круто йти вгору
    if xi <= x[0]: return b_coeff[0]
    if xi >= x[-1]:
        idx = n - 2
    else:
        idx = np.searchsorted(x, xi) - 1
    dx = xi - x[idx]
    return b_coeff[idx] + 2 * c_coeff[idx] * dx + 3 * d_coeff[idx] * dx ** 2

def get_grad_spline(xi):
    idx = 0
    if xi >= x[-1]: idx = n - 2
    else:
        for i in range(len(x)-1):
            if xi >= x[i] and xi <= x[i+1]:
                idx = i
                break
    dx = xi - x[idx]
    res = b_coeff[idx] + 2*c_coeff[idx]*dx + 3*d_coeff[idx]*(dx**2)
    return res * 100

# Генерація точок для плавних графіків створюємо дуже детальну сітку
xx = np.linspace(x[0], x[-1], 500)
yy = np.array([spline_eval(val) for val in xx])
grad = np.array([spline_derivative(val) for val in xx]) * 100
# Розрахунок наборів та спусків
total_ascent = sum(max(y[i] - y[i - 1], 0) for i in range(1, n))
total_descent = sum(abs(min(y[i] - y[i - 1], 0)) for i in range(1, n))

# Кумулятивна енергія розрахунок енергії (якщо людина важить 80 кг)
mass, g = 80, 9.81
work_j = sum(mass * g * max(yy[i] - yy[i-1], 0) for i in range(1, len(yy)))
work_kcal = work_j * 0.000239006
energy = [0]
for i in range(1, len(yy)):
    dh = yy[i] - yy[i - 1]
    # Додаємо енергію тільки коли йдемо вгору
    energy.append(energy[-1] + (mass * g * dh if dh > 0 else 0))
energy = np.array(energy)


# Розрахунок похибок для 10, 15, 20 вузлів порівняно з 21
def get_error_stats(node_count):
    indices = np.linspace(0, len(x) - 1, node_count, dtype=int)
    x_sub, y_sub = x[indices], y[indices]
    # Локальний розрахунок для похибки
    h_s = np.diff(x_sub);
    n_s = len(x_sub)
    As = np.zeros(n_s);
    Bs = np.zeros(n_s);
    Cs = np.zeros(n_s);
    Ds = np.zeros(n_s)
    Bs[0], Bs[-1] = 1, 1
    for i in range(1, n_s - 1):
        As[i] = h_s[i - 1];
        Bs[i] = 2 * (h_s[i - 1] + h_s[i]);
        Cs[i] = h_s[i]
        Ds[i] = 6 * ((y_sub[i + 1] - y_sub[i]) / h_s[i] - (y_sub[i] - y_sub[i - 1]) / h_s[i - 1])
    Bp = Bs.copy();
    Dp = Ds.copy()
    for i in range(1, n_s):
        m = As[i] / Bp[i - 1];
        Bp[i] -= m * Cs[i - 1];
        Dp[i] -= m * Dp[i - 1]
    Ms = np.zeros(n_s);
    Ms[-1] = Dp[-1] / Bp[-1]
    for i in range(n_s - 2, -1, -1): Ms[i] = (Dp[i] - Cs[i] * Ms[i + 1]) / Bp[i]
    acs = y_sub[:-1];
    ccs = Ms[:-1] / 2;
    dcs = np.diff(Ms) / (6 * h_s)
    bcs = (np.diff(y_sub) / h_s) - (h_s * (2 * Ms[:-1] + Ms[1:]) / 6)

    def eval_s(xi):
        idx = np.searchsorted(x_sub, xi) - 1
        idx = max(0, min(idx, n_s - 2))
        dx = xi - x_sub[idx]
        return acs[idx] + bcs[idx] * dx + ccs[idx] * dx ** 2 + dcs[idx] * dx ** 3

    errors = np.abs(np.array([eval_s(xi) for xi in x]) - y)
    return errors, np.sum(errors)

#Додаткові функції для аналізу (уніфіковані змінні)

def get_spline_for_nodes(x_nodes, y_nodes):
    """Універсальна функція для розрахунку параметрів сплайна для будь-якої кількості вузлів"""
    n_m = len(x_nodes)
    h_m = np.diff(x_nodes)
    A_m, B_m, C_m, D_m = np.zeros(n_m), np.zeros(n_m), np.zeros(n_m), np.zeros(n_m)
    B_m[0], B_m[-1] = 1, 1
    for i in range(1, n_m - 1):
        A_m[i] = h_m[i - 1]
        B_m[i] = 2 * (h_m[i - 1] + h_m[i])
        C_m[i] = h_m[i]
        D_m[i] = 6 * ((y_nodes[i + 1] - y_nodes[i]) / h_m[i] - (y_nodes[i] - y_nodes[i - 1]) / h_m[i - 1])

    # Метод прогонки
    for i in range(1, n_m):
        m = A_m[i] / B_m[i - 1]
        B_m[i] -= m * C_m[i - 1]
        D_m[i] -= m * D_m[i - 1]

    M_m = np.zeros(n_m)
    M_m[-1] = D_m[-1] / B_m[-1]
    for i in range(n_m - 2, -1, -1):
        M_m[i] = (D_m[i] - C_m[i] * M_m[i + 1]) / B_m[i]

    return x_nodes, y_nodes, h_m, M_m


def eval_custom_spline(xi, x_n, y_n, h_n, M_n):
    """Обчислення значення сплайна в довільній точці xi"""
    if xi <= x_n[0]: return y_n[0]
    if xi >= x_n[-1]: return y_n[-1]
    idx = np.searchsorted(x_n, xi) - 1
    dx = xi - x_n[idx]
    a = y_n[idx]
    c = M_n[idx] / 2
    d = (M_n[idx + 1] - M_n[idx]) / (6 * h_n[idx])
    b = (y_n[idx + 1] - y_n[idx]) / h_n[idx] - (h_n[idx] * (2 * M_n[idx] + M_n[idx + 1]) / 6)
    return a + b * dx + c * dx ** 2 + d * dx ** 3



# ВИВІД

print("--- ЗАГАЛЬНА ІНФОРМАЦІЯ ---")
print("Кількість точок (вузлів):", n)
print("Загальна відстань маршруту:", round(x[-1], 2), "метрів")
print("Сумарний підйом:", round(total_ascent, 1), "м")
print("Сумарний спуск:", round(total_descent, 1), "м")
print("")

print("--- ТАБЛИЦЯ КООРДИНАТ ТА ВИСОТ ---")
print("№ | Широта | Довгота | Висота (м)")
for i in range(len(results)):
    p = results[i]
    # Просто виводимо через кому, без складних відступів
    print(i + 1, "|", p['latitude'], "|", p['longitude'], "|", p['elevation'])

print("\n--- ВІДСТАНЬ І ВИСОТА ---")
for i in range(n):
    print("Точка", i + 1, ": відстань =", round(x[i], 2), "м, висота =", y[i], "м")

print("\n--- КОЕФІЦІЄНТИ МАТРИЦІ (Метод прогонки) ---")
for i in range(n):
    print("Рядок", i, ": A =", round(A[i], 2), "B =", round(B[i], 2), "C =", round(C[i], 2), "D =", round(D[i], 4))

print("\n--- ДРУГІ ПОХІДНІ (M) ---")
for i in range(len(M)):
    print("M[", i, "] =", round(M[i], 6))

print("\n--- КОЕФІЦІЄНТИ СПЛАЙНІВ ПО СЕГМЕНТАХ ---")
for i in range(len(a_coeff)):
    print("Сегмент", i, "(між точками", i, "та", i+1, "):")
    print("  a =", round(a_coeff[i], 3))
    print("  b =", round(b_coeff[i], 3))
    print("  c =", round(c_coeff[i], 3))
    print("  d =", round(d_coeff[i], 3))

print("\n--- АНАЛІЗ КРУТИЗНИ (ГРАДІЄНТ) ---")
# Використаємо простий цикл для перших 10 точок
for i in range(10):
    g_val = get_grad_spline(x[i])
    print("Вузол", i, ": крутизна =", round(g_val, 4), "%")

print("\n--- ПОРІВНЯННЯ ПОХИБОК ---")
for count in [10, 15, 20]:
    errs, total_err = get_error_stats(count)
    print("Для", count, "вузлів:")
    print("  Сумарна похибка:", round(total_err, 4))
    # Виведемо список похибок просто як масив
    print("  Похибки в точках:", errs)

print("\n--- ФІЗИЧНІ ПОКАЗНИКИ ---")
print("Максимальний підйом:", round(np.max(grad), 2), "%")
print("Максимальний спуск:", round(np.min(grad), 2), "%")
print("Середня крутизна:", round(np.mean(np.abs(grad)), 2), "%")
print("Дуже круті ділянки (>15%):", np.sum(np.abs(grad) > 15))
print("Робота в кілоджоулях:", round(work_j / 1000, 2), "кДж")
print("Витрачена енергія:", round(work_j * 0.000239, 2), "ккал")

# Візуалізація


# Графік 1: Маршрут за координатами
lats = [p["latitude"] for p in results]
lons = [p["longitude"] for p in results]
plt.figure(figsize=(8, 6))
plt.plot(lons, lats, 'o-')
plt.title("Маршрут за GPS координатами")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.show()

# Підготовка даних для порівняння
node_counts = [10, 15, 20]
colors = {10: 'orange', 15: 'green', 20: 'red'}
error_colors = {10: 'tab:blue', 15: 'tab:orange', 20: 'tab:green'}
xx_comp = np.linspace(x[0], x[-1], 500)
yy_ref = np.array([spline_eval(val) for val in xx_comp])

# Графік 2: Вплив кількості вузлів
plt.figure(figsize=(10, 6))
plt.plot(xx_comp, yy_ref, label="21 вузол (еталон)", linewidth=2, zorder=5)

comparison_results = {}

for count in node_counts:
    # Вибираємо вузли рівномірно
    indices = np.linspace(0, len(x) - 1, count, dtype=int)
    xn, yn, hn, Mn = get_spline_for_nodes(x[indices], y[indices])

    yy_sub = np.array([eval_custom_spline(val, xn, yn, hn, Mn) for val in xx_comp])
    comparison_results[count] = yy_sub

    plt.plot(xx_comp, yy_sub, label=f"{count} вузлів", color=colors[count])

plt.title("Вплив кількості вузлів")
plt.legend()
plt.grid(True)
plt.show()

# Графік : Похибка апроксимації
plt.figure(figsize=(10, 6))
for count in node_counts:
    # Абсолютна різниця між еталоном (21) та поточною кількістю вузлів
    error = np.abs(yy_ref - comparison_results[count])
    plt.plot(xx_comp, error, label=f"{count} вузлів", color=error_colors[count])

plt.title("Похибка апроксимації")
plt.legend()
plt.grid(True)
plt.show()

# Графік : Висота маршруту від кумулятивної відстані
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o-', color='green', markersize=6)
plt.title("Висота маршруту від кумулятивної відстані")
plt.xlabel("Кумулятивна відстань (м)")
plt.ylabel("Висота (м)")
plt.grid(True)
plt.show()

# Графік : Профіль висоти
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', label="GPS точки")
plt.plot(xx, yy, label="Кубічний сплайн", linewidth=2)
plt.title("Профіль висоти")
plt.xlabel("Відстань (м)")
plt.ylabel("Висота (м)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()

# Графік : Градієнт
plt.figure(figsize=(10, 6))
plt.plot(xx, grad, color='tab:blue')
plt.title("Градієнт (%)")
plt.xlabel("Відстань (м)")
plt.ylabel("Крутизна (%)")
plt.grid(True)
plt.show()

# Графік : Енергія
plt.figure(figsize=(10, 6))
plt.plot(xx, energy, color='tab:red')
plt.title("Кумулятивна енергія (Дж)")
plt.xlabel("Відстань (м)")
plt.ylabel("Енергія (Дж)")
plt.grid(True)
plt.show()


