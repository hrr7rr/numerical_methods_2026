import math
import numpy as np
import matplotlib.pyplot as plt

# Функція і аналітична похідна
def M(t):
    return 50 * math.exp(-0.1 * t) + 5 * math.sin(t)

def M_der_exact(t):
    return -5 * math.exp(-0.1 * t) + 5 * math.cos(t)

x0 = 1.0
exact = M_der_exact(x0)
print("Явний вираз похідної:")
print("M'(t) = -5e^(-0.1t) + 5cos(t)")
print(f"Точне значення при x0={x0}: M'(x0) = {exact}\n")

# Центральна різниця
def central_diff(h):
    return (M(x0 + h) - M(x0 - h)) / (2*h)

# Дослідження похибки
print("===== ДОСЛІДЖЕННЯ ПОХИБКИ =====")
h_values = [10**(-k) for k in range(1, 16)]
errors = []

best_h = None
min_error = 1e10

for h in h_values:
    approx = central_diff(h)
    error = abs(approx - exact)
    errors.append(error)
    print(f"h = {h:.1e}  y'(h) = {approx:.10f}   R = {error:.10e}")
    if error < min_error:
        min_error = error
        best_h = h

print(f"\nОптимальний h0 = {best_h}")
print(f"Досягнута точність R0 = {min_error}\n")

# Обчислення двох кроків для уточнення
h = 1e-3
y_h = central_diff(h)
y_2h = (M(x0 + 2*h) - M(x0 - 2*h)) / (4*h)

R1 = abs(y_h - exact)

print("===== ДВА КРОКИ =====")
print(f"y'(h) = {y_h}")
print(f"y'(2h) = {y_2h}")
print(f"Похибка R1 = {R1}\n")

# Метод Рунге–Ромберга
y_RR = y_h + (y_h - y_2h)/3
R2 = abs(y_RR - exact)

print("===== МЕТОД РУНГЕ–РОМБЕРГА =====")
print(f"Уточнене значення y'_R = {y_RR}")
print(f"Похибка R2 = {R2}")
print("Похибка зменшилась (порядок підвищився)." if R2 < R1 else "Похибка не зменшилась.")

# Метод Ейткена
y_4h = (M(x0 + 4*h) - M(x0 - 4*h)) / (8*h)
y_E = (y_2h**2 - y_4h*y_h) / (2*y_2h - y_4h - y_h)
p = (1/math.log(2)) * math.log(abs((y_4h - y_2h)/(y_2h - y_h)))
R3 = abs(y_E - exact)

print("\n===== МЕТОД ЕЙТКЕНА =====")
print(f"y'(h) = {y_h}")
print(f"y'(2h) = {y_2h}")
print(f"y'(4h) = {y_4h}")
print(f"Уточнене значення y'_E = {y_E}")
print(f"Порядок точності p ≈ {p}")
print(f"Похибка R3 = {R3}\n")

# Графік похибки
plt.figure()
plt.loglog(h_values, errors, marker='o')
plt.xlabel("Крок h")
plt.ylabel("Похибка")
plt.title("Залежність похибки від кроку")
plt.grid(True)
plt.show()

# Висновок щодо поливу
if abs(exact) < 1:
    print("Швидкість зміни вологості мала — полив можна не вмикати.")
elif exact < 0:
    print("Вологість швидко зменшується — необхідно вмикати полив.")
else:
    print("Вологість зростає — полив не потрібен.")