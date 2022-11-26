import pickle
import numpy as np 
from sklearn import model_selection, linear_model, preprocessing
import matplotlib.pyplot as plt

def sqr(x):
    return x*x

with open("./Q1_data/data.pkl", "rb") as file:
    data = pickle.load(file)
report = open("./Q1_data/q1_result.md", "w")
print("Polynomial Degree | Bias | Variance", file=report)
print(" - | - | -", file=report)

x_list = []
y_list = []
for coord in data:
    x_list.append(coord[0])
    y_list.append(coord[1])
    # print(coord[0])

x_list = np.array(x_list)
y_list = np.array(y_list)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x_list, y_list, test_size=0.1)
x_train = x_train.reshape(-1, 1)
# y_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)
# y_test = y_test.reshape(-1, 1)
# print(y_test)
# x_part_train = np.array([])
x_part_train = np.array(np.array_split(x_train, 10))
y_part_train = np.array(np.array_split(y_train, 10))

avg_bias_2 = np.zeros(10)
avg_var = np.zeros(10)
# print(x_part_train.shape)
for poly_degree in range(1, 10):
    plt.scatter(x_test, y_test, color="black")
    pol = preprocessing.PolynomialFeatures(poly_degree)
    y_expect = np.zeros(500)
    # y_expect = np.array(np.array_split(y_expect, 10))
    y_predict = np.zeros(5000)
    y_predict = np.array(np.array_split(y_predict, 10))
    bias_2 = np.zeros(500)
    for i in range(10):

        x_poly_train = np.array(pol.fit_transform(x_part_train[i]))
        x_poly_test = np.array(pol.fit_transform(x_test))
        # print(x_poly_test)
        # np.resize(x_poly_train, 500)
        # np.resize(x_poly_test, 500)
        # print(len(x_poly_test) - len(y_test))
        reg = linear_model.LinearRegression()
        # reg.fit(x_part_train[i], y_part_train[i])
        reg.fit(x_poly_train, y_part_train[i])
        y_predict[i] = reg.predict(x_poly_test)
        for j in range(500):
            y_expect[j] += y_predict[i][j]
        # plt.scatter(x_poly_test, y_test, color="black")
        # plt.scatter(x_part_train[i], y_part_train[i])
        # plt.plot(x_test, y_predict, color="blue")
        # print(reg.coef_)
        # plt.show()
    
    for j in range(500):
        y_expect[j] = y_expect[j] / 10
        bias_2[j] = sqr(y_expect[j] - y_test[j])
        avg_bias_2[poly_degree] += bias_2[j]

    var = np.zeros(500)
    avg_var = 0
    for j in range(500):
        for i in range(10):
            var[j] += sqr(y_predict[i][j] - y_expect[j])
        var[j] = var[j] / 10
        avg_var += var[j]
    avg_var /= 500
    # print(avg_var, end=" | ")
    avg_bias_2[poly_degree] /= 500
    # print(avg_bias_2[poly_degree])

    print(poly_degree, " | ", avg_bias_2[poly_degree], " | ", avg_var, file=report)

    plt.scatter(x_test, y_expect)
    # plt.show()

report.close()