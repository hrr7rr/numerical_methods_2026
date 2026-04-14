import numpy as np
import matplotlib.pyplot as plt


#  дані
def prepare_files(n=100):
    A = np.random.uniform(1, 10, (n, n))
    x_exact = np.full(n, 2.5)
    b = A @ x_exact

    np.savetxt('matrix_A.txt', A)
    np.savetxt('vector_B.txt', b)

    return A, b, x_exact


# LU-розклад
def lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.eye(n)

    for k in range(n):
        for i in range(k, n):
            L[i, k] = A[i, k] - np.dot(L[i, :k], U[:k, k])

        for i in range(k + 1, n):
            U[k, i] = (A[k, i] - np.dot(L[k, :k], U[:k, i])) / L[k, k]

    return L, U


# Розв'язання
def solve_system(L, U, b):
    n = len(L)

    # прямий хід
    z = np.zeros(n)
    for k in range(n):
        z[k] = (b[k] - np.dot(L[k, :k], z[:k])) / L[k, k]

    # зворотний хід
    x = np.zeros(n)
    for k in range(n - 1, -1, -1):
        x[k] = z[k] - np.dot(U[k, k + 1:], x[k + 1:])

    return x


#  Похибка
def get_eps(A, x, b):
    return np.max(np.abs(A @ x - b))


# основна програма

# Генерація
A, b, x_exact = prepare_files(100)

# Зчитування
A = np.loadtxt('matrix_A.txt')
b = np.loadtxt('vector_B.txt')

print("Матриця A (перші 3 рядки):\n", A[:3])
print("\nВектор b (перші 5 елементів):\n", b[:5])

# LU
L, U = lu_decomposition(A)

print("\nМатриця L (перші 3 рядки):\n", L[:3])
print("\nМатриця U (перші 3 рядки):\n", U[:3])

np.savetxt('matrix_L.txt', L)
np.savetxt('matrix_U.txt', U)

# Початковий розв’язок
x_0 = solve_system(L, U, b)

print("\nПочатковий розв'язок x0 (перші 5):\n", x_0[:5])

# Початкова похибка
eps_0 = get_eps(A, x_0, b)
print(f"\nПочаткова похибка: {eps_0:.2e}")

# CLAR
x_refined = x_0.copy()
target_eps = 1e-14
history = [eps_0]
iters = 0

while history[-1] > target_eps and iters < 50:

    r = b - A @ x_refined
    delta_x = solve_system(L, U, r)

    # зупинка по машинній точності
    if np.linalg.norm(delta_x) < 1e-16:
        print("\nДосягнута межа машинної точності")
        break

    x_refined = x_refined + delta_x

    current_eps = get_eps(A, x_refined, b)
    history.append(current_eps)
    iters += 1

    print(f"Ітерація {iters}: eps = {current_eps:.2e}")

# результати
print("\nФінальний розв'язок (перші 5):\n", x_refined[:5])

print(f"\nФінальна похибка: {history[-1]:.2e}")

# похибка до істинного розв’язку
true_error = np.linalg.norm(x_refined - x_exact)
print(f"Похибка до істинного розв’язку: {true_error:.2e}")

# норма вектора
norm_x = np.linalg.norm(x_refined)
print(f"Норма ||x||: {norm_x:.2e}")

print(f"\nКількість ітерацій: {iters}")


