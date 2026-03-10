import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# 1. Зчитування CSV
# ----------------------------
def read_csv(file):
    data = pd.read_csv(file)
    x = data["Month"].values
    y = data["Temp"].values
    return x, y


# ---------------------------
# 2. Формування матриці
# ----------------------------
def form_matrix(x, m):

    n = len(x)
    A = np.zeros((m+1, m+1))

    for i in range(m+1):
        for j in range(m+1):
            A[i,j] = sum(x[k]**(i+j) for k in range(n))

    return A


# ----------------------------
# 3. Формування вектора
# ----------------------------
def form_vector(x, y, m):

    n = len(x)
    b = np.zeros(m+1)

    for i in range(m+1):
        b[i] = sum(y[k]*(x[k]**i) for k in range(n))

    return b


# ----------------------------
# 4. Метод Гауса
# ----------------------------
def gauss(A, b):

    n = len(b)

    A = A.astype(float)
    b = b.astype(float)

    for k in range(n):

        max_row = max(range(k,n), key=lambda i: abs(A[i,k]))
        A[[k,max_row]] = A[[max_row,k]]
        b[k], b[max_row] = b[max_row], b[k]

        for i in range(k+1, n):

            factor = A[i,k] / A[k,k]

            for j in range(k, n):
                A[i,j] -= factor * A[k,j]

            b[i] -= factor * b[k]

    x = np.zeros(n)

    for i in range(n-1,-1,-1):

        s = sum(A[i,j]*x[j] for j in range(i+1,n))
        x[i] = (b[i]-s)/A[i,i]

    return x


# ----------------------------
# 5. Поліном
# ----------------------------
def polynomial(x, coef):

    y = np.zeros_like(x, dtype=float)

    for i in range(len(coef)):
        y += coef[i]*x**i

    return y


# ----------------------------
# 6. Дисперсія
# ----------------------------
def variance(y, y2):

    return np.mean((y-y2)**2)


# ----------------------------
# Основна програма
# ----------------------------
x, y = read_csv("temperature.csv")

max_m = 10
variances = []
coef_list = []

print("Дисперсії для різних степенів полінома:")

for m in range(1, max_m+1):

    A = form_matrix(x,m)
    b = form_vector(x,y,m)

    coef = gauss(A,b)

    y_approx = polynomial(x,coef)

    var = variance(y,y_approx)

    variances.append(var)
    coef_list.append(coef)

    print(f"m={m}  variance={var:.4f}")


# ----------------------------
# оптимальний степінь
# ----------------------------
optimal_m = np.argmin(variances)+1
print("\n2. Аналіз та вибір оптимального ступеня")

min_var = min(variances)

print(f"Мінімальна дисперсія δ = {min_var:.6f}")
print(f"Оптимальний степінь m = {optimal_m}")



coef = coef_list[optimal_m-1]

print("\n3. Коефіцієнти апроксимуючого многочлена:")

for i,c in enumerate(coef):
    print(f"a{i} = {c:.6f}")

y_approx = polynomial(x,coef)


# ----------------------------
# прогноз
# ----------------------------
x_future = np.array([25,26,27])
y_future = polynomial(x_future,coef)

print("\nПрогноз температур:")

print(f"Місяць 25: {y_future[0]:.2f}")
print(f"Місяць 26: {y_future[1]:.2f}")
print(f"Місяць 27: {y_future[2]:.2f}")


# ----------------------------
# Екстраполяція: прогноз температури
# ----------------------------


x_last = x[-5:]
y_last = y[-5:]

plt.figure()

# фактичні дані
plt.scatter(x_last, y_last, color='blue', label="Останні дані")

# прогноз
plt.plot(x_future, y_future, 'g--s', label="Прогноз (наступні 3 міс.)")

plt.xlabel("Місяць")
plt.ylabel("Температура (°C)")

plt.title(" Екстраполяція: прогноз температури")

plt.legend()
plt.grid(True)

plt.show()

# ----------------------------
# 4. Порівняння апроксимацій для різних степенів (m=1..4)
# ----------------------------

plt.figure(figsize=(10,8))

for m in range(1,5):

    coef = coef_list[m-1]

    # густі точки для гладкої кривої
    x_dense = np.linspace(min(x), max(x), 200)
    y_poly = polynomial(x_dense, coef)

    plt.subplot(2,2,m)

    # фактичні дані
    plt.scatter(x, y, color='blue', label="Фактичні дані")

    # поліном
    plt.plot(x_dense, y_poly, color='red', label=f"Поліном m={m}")

    plt.title(f"Степінь m={m} (Дисперсія: {variances[m-1]:.2f})")
    plt.xlabel("Місяць")
    plt.ylabel("Температура (°C)")

    plt.legend()
    plt.grid()

plt.suptitle("4. Порівняння апроксимацій для різних степенів (m=1..4)")

plt.tight_layout()

plt.show()


# ----------------------------
# графік апроксимації
# ----------------------------
plt.figure()

plt.scatter(x,y,label="Фактичні дані")
plt.plot(x,y_approx,label="Апроксимація")

plt.xlabel("Місяць")
plt.ylabel("Температура")

plt.title("Метод найменших квадратів")

plt.legend()
plt.grid()

plt.show()


# ----------------------------
# графік дисперсії
# ----------------------------
plt.figure()

plt.plot(range(1,max_m+1), variances, marker='o')

plt.xlabel("Степінь полінома m")
plt.ylabel("Дисперсія")

plt.title("Залежність дисперсії від степеня")

plt.grid()

plt.show()


# ----------------------------
# табуляція похибки
# ----------------------------
x_dense = np.linspace(x[0], x[-1], 21)

plt.figure()

for m in range(1, max_m+1):

    coef = coef_list[m-1]

    y_poly = polynomial(x_dense,coef)

    y_real = np.interp(x_dense, x, y)

    error = y_real - y_poly

    plt.plot(x_dense,error,label=f"m={m}")

plt.xlabel("x")
plt.ylabel("Похибка")

plt.title("Графіки похибки")

plt.legend()
plt.grid()

plt.show()
# ----------------------------
# графік похибки для оптимального полінома
# ----------------------------

error_opt = y - y_approx

# ----------------------------
# Табуляція похибки
# ----------------------------

print("\n5. Табуляція похибки апроксимації:\n")

print("Місяць\tРеальна T\tАпроксимація\tПохибка ε(x)")

for i in range(len(x)):
    print(f"{x[i]:2d}\t{y[i]:8.2f}\t{y_approx[i]:12.2f}\t{error_opt[i]:10.4f}")

plt.figure()

plt.plot(x, error_opt, marker='o')

plt.xlabel("Місяць")
plt.ylabel("Похибка")

plt.title("Похибка апроксимації")

plt.grid()

plt.show()