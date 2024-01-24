from sklearn import datasets
import random

from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

dimension_data = [(random.random() * 0.3, random.random() * 100) for i in range(100)]


def generate_data(count):
    X = []
    y = []
    for i in range(count):
        y_data = random.random()
        x_data = []
        for dim in dimension_data:
            x_data.append((y_data + (random.random() - 0.5) * dim[0]) * dim[1])
        X.append(x_data)
        y.append(y_data)
    return X, y

X, y = generate_data(1000)  # DO NOT PUT 1 MILLION, IT WILL MAKE YOUR COMPUTER CRASH

scaler = StandardScaler()
X = scaler.fit_transform(X)


X_test, y_test = generate_data(1000)
X_test = scaler.transform(X_test)

kernels = ["linear", "polynomial", "rbf"]
for kernel in kernels:
    alpha_values = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
    model = GridSearchCV(KernelRidge(kernel=kernel), alpha_values, scoring="neg_mean_squared_error", cv=5)
    model.fit(X, y)
    model = model.best_estimator_
    y_predict = model.predict(X_test)
    print("SCR pour le kernel " + kernel + " : " + str(mean_squared_error(y_test, y_predict) * len(y_test)))
