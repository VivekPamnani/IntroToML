import pickle
import numpy as np 
from sklearn import model_selection, linear_model, preprocessing
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

Bias_2 = mpatches.Patch(color='red', label='Bias^2')
Variance = mpatches.Patch(color='violet', label='Variance')
Total_error = mpatches.Patch(color='black', label='Total Error\n(+e)')
plt.legend(handles=[Bias_2, Variance, Total_error])

def sqr(x):
    return x*x

with open("./Q2_data/X_train.pkl", "rb") as file:
    X_train = np.array(pickle.load(file))
    x_train = np.reshape(X_train, (20,400))
with open("./Q2_data/Y_train.pkl", "rb") as file:
    Y_train = np.array(pickle.load(file))
    y_train = np.reshape(Y_train, (20,400))
with open("./Q2_data/X_test.pkl", "rb") as file:
    X_test = np.array(pickle.load(file))
    x_test = X_test
with open("./Q2_data/Fx_test.pkl", "rb") as file:
    Fx_test = np.array(pickle.load(file))
    y_test = Fx_test

report = open("./Q2_data/q2_result.md", "w")
print("Polynomial Degree | Bias | Variance", file=report)
print(" - | - | -", file=report)

# for i in y_test:
#     print(i)
# print()
# print(len(y_test))
# quit()

# for i in range(20):
#     x_train[i] = np.array(np.reshape(x_train[i], 400))
#     y_train[i] = np.array(np.reshape(y_train[i], 400))
# x_test = []
# y_test = []
# x_train = []
# y_train = []
# for i in range(20):
#     for j in range(400):
#         x_test.append(X_test[i][j])

# print(len(x_test))
# quit()

avg_bias_2 = np.zeros(10)
avg_var = np.zeros(10)
# print(x_part_train.shape)
for poly_degree in range(1, 10):
    # plt.scatter(x_test, y_test, color="black")
    pol = preprocessing.PolynomialFeatures(poly_degree)
    y_expect = np.zeros(80)
    # y_expect = np.array(np.array_split(y_expect, 10))
    y_predict = np.zeros(1600)
    y_predict = np.array(np.array_split(y_predict, 20))
    bias_2 = np.zeros(80)
    for i in range(20):
        train_x = x_train[i]
        train_x = train_x.reshape(-1,1)
        test_x = x_test
        test_x = test_x.reshape(-1,1)
        x_poly_train = np.array(pol.fit_transform(train_x))
        x_poly_test = np.array(pol.fit_transform(test_x))
        # print(x_poly_test)
        # np.resize(x_poly_train, 500)
        # np.resize(x_poly_test, 500)
        # print(len(x_poly_test) - len(y_test))
        reg = linear_model.LinearRegression()
        # reg.fit(x_part_train[i], y_part_train[i])
        reg.fit(x_poly_train, y_train[i])
        y_predict[i] = reg.predict(x_poly_test)
        for j in range(80):
            y_expect[j] += y_predict[i][j]
        # plt.scatter(x_poly_test, y_test, color="black")
        # plt.scatter(x_part_train[i], y_part_train[i])
        # plt.plot(x_test, y_predict, color="blue")
        # print(reg.coef_)
        # plt.show()
    
    for j in range(80):
        y_expect[j] = y_expect[j] / 20
        bias_2[j] = sqr(y_expect[j] - y_test[j])
        avg_bias_2[poly_degree] += bias_2[j]

    var = np.zeros(80)
    for j in range(80):
        for i in range(20):
            var[j] += sqr(y_predict[i][j] - y_expect[j])
        var[j] = var[j] / 20
        avg_var[poly_degree] += var[j]
    avg_var[poly_degree] /= 80
    # print(avg_var[poly_degree], end=" | ")
    avg_bias_2[poly_degree] /= 80
    # print(avg_bias_2[poly_degree])

    print(poly_degree, " | ", avg_bias_2[poly_degree], " | ", avg_var[poly_degree], file=report)

    # plt.scatter(test_x, y_expect)
    # plt.show()

plt.plot(np.arange(1,10), avg_var[1:10], color="violet", antialiased=1)
plt.plot(np.arange(1,10), avg_bias_2[1:10], color="red", antialiased=1)
plt.plot(np.arange(1,9), np.sum([avg_var, avg_bias_2], axis=0)[1:9], color="black", antialiased=1)
plt.xlabel('Complexity')
plt.ylabel('Error')
# plt.savefig('./Q2_data/bias_variance.png')
plt.show()
report.close()