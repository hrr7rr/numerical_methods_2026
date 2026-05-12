import numpy as np
import matplotlib.pyplot as plt

# функція розенброка
def rosenbrock(x):
    x1, x2 = x
    return 100 * (x1**2 - x2)**2 + (x1 - 1)**2

# система нелінійних рівнянь (m=2)
# x² + y² - 4 = 0
# x - y - 1 = 0
def eq1(x):
    return x[0]**2 + x[1]**2 - 4

def eq2(x):
    return x[0] - x[1] - 1

def objective(x):
    return eq1(x)**2 + eq2(x)**2


# метод хука-дживса

def exploratory_search(f, base, step):
    x = np.copy(base)

    for i in range(len(x)):
        f0 = f(x)

        x[i] += step[i]
        if f(x) < f0:
            continue

        x[i] -= 2 * step[i]
        if f(x) < f0:
            continue

        x[i] += step[i]

    return x


def hooke_jeeves(f, x0, step=0.5, alpha=2, eps=1e-6, max_iter=1000):
    n = len(x0)
    step = np.ones(n) * step

    xb = np.array(x0, dtype=float)
    xp = np.copy(xb)

    trajectory = [np.copy(xb)]

    iterations = 0

    while np.max(step) > eps and iterations < max_iter:
        xn = exploratory_search(f, xp, step)

        if f(xn) < f(xb):
            xp = xn + alpha * (xn - xb)
            xb = xn
        else:
            step /= 2
            xp = xb

        trajectory.append(np.copy(xb))
        iterations += 1

    return xb, f(xb), np.array(trajectory), iterations


print("    Функція Розенброка")

x_min, f_min, traj_rosen, steps_rosen = hooke_jeeves(
    rosenbrock,
    x0=[-1.2, 0],
    step=0.5
)

print("    Мінімум:")
print("x1 =", x_min[0])
print("x2 =", x_min[1])
print("Значення функції f(x) =", f_min)
print("Кількість кроків =", steps_rosen)

# розв'язок системи

print("\n             Система нелінійних рівнянь")

solution, val, trajectory, steps = hooke_jeeves(
    objective,
    x0=[1, 1],
    step=0.5
)

print("Розв’язок:")
print("x =", solution[0])
print("y =", solution[1])
print("Значення цільової функції F(x,y) =", val)
print("Кількість кроків =", steps)

# запис у файл

with open("trajectory.txt", "w") as f:
    for point in trajectory:
        f.write(f"{point[0]} {point[1]}\n")

print("\nФайл trajectory.txt створено")


# графіки рівнянь

x = np.linspace(-3, 3, 400)
y = np.linspace(-3, 3, 400)

X, Y = np.meshgrid(x, y)

Z1 = X**2 + Y**2 - 4
Z2 = X - Y - 1

plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z1, levels=[0], colors='blue')
plt.contour(X, Y, Z2, levels=[0], colors='red')

plt.plot(solution[0], solution[1], 'ko', label='Розв’язок')

plt.title("Система нелінійних рівнянь")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

# цільова функція

Z = Z1**2 + Z2**2

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z)

ax.set_title("Цільова функція")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("F(x,y)")

plt.show()


# траєкторія спуску
plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=30)

plt.plot(
    trajectory[:, 0],
    trajectory[:, 1],
    'ro-',
    label='Траєкторія'
)

plt.title("Траєкторія спуску")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()

plt.show()


# графік функції розенброка

x1 = np.linspace(-2, 2, 400)
x2 = np.linspace(-1, 3, 400)

X1, X2 = np.meshgrid(x1, x2)

Zr = 100 * (X1**2 - X2)**2 + (X1 - 1)**2

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X1, X2, Zr)

ax.set_title("Функція Розенброка")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("f(x)")

plt.show()