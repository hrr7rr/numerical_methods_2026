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
print("Точне значення при x0=1")
print("M'(x0) =", exact)

# Центральна різниця
def central_diff(h):
    return (M(x0 + h) - M(x0 - h)) / (2*h)

# Дослідження похибки
print("\nДослідження похибки")
best_h = None
min_error = 1e10
h_values = [10**(-k) for k in range(1, 16)]
errors = []

for h in h_values:
    approx = central_diff(h)
    R = abs(approx - exact)
    errors.append(R)
    print(f"h = 1e-{int(-math.log10(h)):<2}  y'(h) = {approx:.10f}   R = {R:.10e}")
    if R < min_error:
        min_error = R
        best_h = h

print("\nОптимальний h0 =", best_h)
print("Досягнута точність R0 =", min_error)

# h = 10^-3
h = 1e-3
print("\nh = 10^-3")

# Два кроки h та 2h
y_h = central_diff(h)
y_2h = (M(x0 + 2*h) - M(x0 - 2*h)) / (4*h)

print("\nОбчислення двох кроків")
print("y'(h)  =", y_h)
print("y'(2h) =", y_2h)

# Похибка R1
R1 = abs(y_h - exact)
print("\nПохибка R1")
print("R1 =", R1)

# Метод Рунге–Ромберга
y_RR = y_h + (y_h - y_2h) / 3
R2 = abs(y_RR - exact)

print("\nМетод Рунге–Ромберга")
print("Уточнене значення y'_R =", y_RR)
print("Похибка R2 =", R2)
print("Похибка зменшилась (порядок підвищився)." if R2 < R1 else "Похибка не зменшилась.")

# Метод Ейткена
y_4h = (M(x0 + 4*h) - M(x0 - 4*h)) / (8*h)
y_E = (y_2h**2 - y_4h*y_h) / (2*y_2h - y_4h - y_h)
p = (1/math.log(2)) * math.log(abs((y_4h - y_2h)/(y_2h - y_h)))
R3 = abs(y_E - exact)

print("\nМетод Ейткена")
print("y'(h)   =", y_h)
print("y'(2h)  =", y_2h)
print("y'(4h)  =", y_4h)
print("Уточнене значення y'_E =", y_E)
print("Порядок точності p ≈", p)
print("Похибка R3 =", R3)

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