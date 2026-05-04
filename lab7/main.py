import numpy as np
import matplotlib.pyplot as plt


# Генерація матриці A

def generate_matrix(n):
    A = np.random.rand(n, n)

    # забезпечення діагонального переважання
    for i in range(n):
        A[i, i] = sum(abs(A[i])) + 1

    return A


#  Запис/читання файлів
def save_matrix(A, filename):
    np.savetxt(filename, A)


def load_matrix(filename):
    return np.loadtxt(filename)


def save_vector(b, filename):
    np.savetxt(filename, b)


def load_vector(filename):
    return np.loadtxt(filename)


#  Норми
def vector_norm(x):
    return np.linalg.norm(x, ord=np.inf)

def matrix_norm(A):
    return np.linalg.norm(A, ord=np.inf)


#  Метод простої ітерації
def simple_iteration(A, b, x0, eps, max_iter=10000):
    n = len(b)
    tau = 1 / matrix_norm(A)

    x = x0.copy()
    history = []

    for k in range(max_iter):
        x_new = x - tau * (A @ x - b)

        err = vector_norm(x_new - x)
        history.append(err)

        if err < eps:
            return x_new, k + 1, history

        x = x_new

    return x, max_iter, history


#  Метод Якобі
def jacobi(A, b, x0, eps, max_iter=10000):
    n = len(b)
    x = x0.copy()
    history = []

    D = np.diag(A)
    R = A - np.diagflat(D)

    for k in range(max_iter):
        x_new = (b - R @ x) / D

        err = vector_norm(x_new - x)
        history.append(err)

        if err < eps:
            return x_new, k + 1, history

        x = x_new

    return x, max_iter, history


#  Метод Зейделя
def seidel(A, b, x0, eps, max_iter=10000):
    n = len(b)
    x = x0.copy()
    history = []

    for k in range(max_iter):
        x_new = x.copy()

        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))

            x_new[i] = (b[i] - s1 - s2) / A[i][i]

        err = vector_norm(x_new - x)
        history.append(err)

        if err < eps:
            return x_new, k + 1, history

        x = x_new

    return x, max_iter, history


#  Головна програма

def main():
    n = 100
    eps = 1e-14

    # Генерація
    A = generate_matrix(n)

    # точний розв'язок
    x_true = np.full(n, 2.5)

    # b = A*x
    b = A @ x_true

    # збереження
    save_matrix(A, "A.txt")
    save_vector(b, "b.txt")

    # зчитування
    A = load_matrix("A.txt")
    b = load_vector("b.txt")

    # початкове наближення
    x0 = np.ones(n)

    # Обчислення
    x_si, it_si, hist_si = simple_iteration(A, b, x0, eps)
    x_j, it_j, hist_j = jacobi(A, b, x0, eps)
    x_s, it_s, hist_s = seidel(A, b, x0, eps)

    def print_results(name, x, iterations, x_true, A, b):
        print(f"\n{name}")
        print("Кількість ітерацій:", iterations)
        print("Перші 5 значень розв’язку:")
        print(x[:5])
        error = np.linalg.norm(x - x_true, ord=np.inf)
        print("Максимальна похибка:", error)

        residual = np.linalg.norm(A @ x - b, ord=np.inf)
        print("Норма нев’язки:", residual)

    print_results("Метод простої ітерації", x_si, it_si, x_true, A, b)
    print_results("Метод Якобі", x_j, it_j, x_true, A, b)
    print_results("Метод Зейделя", x_s, it_s, x_true, A, b)




if __name__ == "__main__":
    main()
