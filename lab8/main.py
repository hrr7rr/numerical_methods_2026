import math
import cmath
import numpy as np
import matplotlib.pyplot as plt

# параметри

EPS = 1e-10
MAX_ITER = 200

A = -10
B = 10
H = 0.1


# трансцендентна функція

def f(x):
    return math.sin(x) - 0.3 * x

def df(x):
    return math.cos(x) - 0.3

def d2f(x):
    return -math.sin(x)


# табуляція

def tabulate():
    roots = []

    with open("table.txt", "w", encoding="utf-8") as file:
        x = A
        prev = f(x)

        while x <= B:
            y = f(x)
            file.write(f"{x:.4f} {y:.8f}\n")

            if prev * y < 0:
                roots.append((x - H, x))

            prev = y
            x += H

    return roots


# проста ітерація

def simple_iteration(x0):
    alpha = 0.5
    x = x0

    for k in range(MAX_ITER):
        x_new = x - alpha * f(x)

        if abs(x_new - x) < EPS and abs(f(x_new)) < EPS:
            return x_new, k + 1

        x = x_new

    return x, MAX_ITER


# ньютон

def newton(x0):
    x = x0

    for k in range(MAX_ITER):
        dfx = df(x)
        if abs(dfx) < EPS:
            return x, k

        x_new = x - f(x) / dfx

        if abs(x_new - x) < EPS:
            return x_new, k + 1

        x = x_new

    return x, MAX_ITER


# чебишев

def chebyshev(x0):
    x = x0

    for k in range(MAX_ITER):
        fx = f(x)
        dfx = df(x)

        if abs(dfx) < EPS:
            return x, k

        x_new = x - fx / dfx - (d2f(x) * fx**2) / (2 * dfx**3)

        if abs(x_new - x) < EPS:
            return x_new, k + 1

        x = x_new

    return x, MAX_ITER


#хорди

def chord(x0, x1):
    for k in range(MAX_ITER):

        f0, f1 = f(x0), f(x1)

        if abs(f1 - f0) < EPS:
            return x1, k

        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)

        if abs(x2 - x1) < EPS:
            return x2, k + 1

        x0, x1 = x1, x2

    return x1, MAX_ITER


# парабола

def parabola(x0, x1, x2):
    for k in range(MAX_ITER):

        f0, f1, f2 = f(x0), f(x1), f(x2)

        h1 = x1 - x0
        h2 = x2 - x1

        if abs(h1) < EPS or abs(h2) < EPS:
            return x2, k

        d1 = (f1 - f0) / h1
        d2 = (f2 - f1) / h2

        a = (d2 - d1) / (h2 + h1)
        b = a * h2 + d2
        c = f2

        D = b*b - 4*a*c

        if D < 0:
            return x2, k

        D = math.sqrt(D)

        dx = -2*c / (b + D) if abs(b + D) > abs(b - D) else -2*c / (b - D)

        x3 = x2 + dx

        if abs(x3 - x2) < EPS:
            return x3, k + 1

        x0, x1, x2 = x1, x2, x3

    return x2, MAX_ITER


# зворотня інтерполяція

def inverse_interpolation(x0, x1):
    for k in range(MAX_ITER):

        f0, f1 = f(x0), f(x1)

        if abs(f1 - f0) < EPS:
            return x1, k

        x2 = (x0*f1 - x1*f0) / (f1 - f0)

        if abs(x2 - x1) < EPS:
            return x2, k + 1

        x0, x1 = x1, x2

    return x1, MAX_ITER


#горнер

def horner(coeffs, x):
    r = coeffs[0]
    for c in coeffs[1:]:
        r = r*x + c
    return r


def horner_derivative(coeffs, x):
    b = coeffs[0]
    db = 0

    for c in coeffs[1:]:
        db = db*x + b
        b = b*x + c

    return db


# ньютон для полінома

def poly_newton(coeffs, x0):
    x = x0

    for k in range(MAX_ITER):

        fx = horner(coeffs, x)
        dfx = horner_derivative(coeffs, x)

        if abs(dfx) < EPS:
            return x, k

        x_new = x - fx/dfx

        if abs(x_new - x) < EPS:
            return x_new, k + 1

        x = x_new

    return x, MAX_ITER


# синтетичне ділення

def synthetic_division(coeffs, root):
    res = [coeffs[0]]

    for i in range(1, len(coeffs)-1):
        res.append(coeffs[i] + res[-1]*root)

    rem = coeffs[-1] + res[-1]*root
    return res, rem


# метод ліма(комплексні корені)

def lina_method(coeffs, root):
    reduced, rem = synthetic_division(coeffs, root)

    if abs(rem) > EPS:
        return None, None

    a, b, c = reduced

    D = b*b - 4*a*c
    sqrtD = cmath.sqrt(D)

    x1 = (-b + sqrtD) / (2*a)
    x2 = (-b - sqrtD) / (2*a)

    return x1, x2


# графік

def plot(coeffs):
    x = np.linspace(-5, 5, 500)
    y = np.polyval(coeffs, x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.axhline(0)
    plt.grid(True)
    plt.title("Графік алгебраїчного многочлена")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

# головна програма

print("   Табуляція")
intervals = tabulate()

print("\nІнтервали:")
for i in intervals:
    print(i)

print("\n             Трансцендентні корені")

for i, (a, b) in enumerate(intervals[:2], 1):

    x0 = (a + b) / 2

    print(f"\nКорінь {i}")

    print("Ітерація:", simple_iteration(x0))
    print("Ньютон:", newton(x0))
    print("Чебишев:", chebyshev(x0))
    print("Хорди:", chord(a, b))
    print("Парабола:", parabola(a, x0, b))
    print("Інтерполяція:", inverse_interpolation(a, b))


coeffs = [1, -2, 4, -8]

plot(coeffs)

root, iters = poly_newton(coeffs, 2)


print("\n       Алгебраїчне рівняння")

print(f"\nДійсний корінь (Ньютон + Горнер): {root:.12f}")
print(f"Кількість ітерацій: {iters}")

r1, r2 = lina_method(coeffs, root)

print("\nКомплексні корені (метод Ліна):")

if r1 is not None:
    print(f"{r1.real:.6f} {'+' if r1.imag >= 0 else '-'} {abs(r1.imag):.6f}i")
    print(f"{r2.real:.6f} {'+' if r2.imag >= 0 else '-'} {abs(r2.imag):.6f}i")
else:
    print("Лін: не вдалося знайти комплексні корені (перевір корінь ділення)")