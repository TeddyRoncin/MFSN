import random

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error


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


X, y = generate_data(10000)  # DO NOT PUT 1 MILLION, IT WILL MAKE YOUR COMPUTER CRASH

alpha_values = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
scaler = StandardScaler()
X = scaler.fit_transform(X)

model = GridSearchCV(Ridge(), alpha_values, scoring="neg_mean_squared_error")
model.fit(X, y)
ridge = model.best_estimator_

model = GridSearchCV(Lasso(), alpha_values, scoring="neg_mean_squared_error")
model.fit(X, y)
lasso = model.best_estimator_

X_test, y_test = generate_data(1000)
X_test = scaler.transform(X_test)

y_predict_ridge = ridge.predict(X)
y_test_predict_ridge = ridge.predict(X_test)

y_predict_lasso = lasso.predict(X)
y_test_predict_lasso = lasso.predict(X_test)

print("SCR ridge")
print("\tTrain : " + str(mean_squared_error(y, y_predict_ridge) * len(y)))
print("\tTest : " + str(mean_squared_error(y_test, y_test_predict_ridge) * len(y_test)))

print("SCR lasso")
print("\tTrain : " + str(mean_squared_error(y, y_predict_lasso) * len(y)))
print("\tTest : " + str(mean_squared_error(y_test, y_test_predict_lasso) * len(y_test)))
