import numpy
import pandas
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

dataFrame = pandas.read_csv("SeoulBikeData.csv", encoding="unicode_escape")

# transform categorical features to numeric features
categorical_fields = ["Date", "Holiday", "Seasons"]
for col in categorical_fields:
    dataFrame[col] = dataFrame[col].astype("category").cat.codes

all_cols = []
for i in range(0, 8760):
    line = []
    # remove days with functioning day = no (0 bikes rented)
    if dataFrame["Functioning Day"][i] == "Yes":
        for column in dataFrame.columns[:13]:
            line.append(dataFrame[column][i])
        all_cols.append(line)

# scale data
scaler = StandardScaler()
scaler.fit(all_cols)
rescaledCols = scaler.transform(all_cols)

Y_total = []
X_total = []
print(type(rescaledCols))
for row in rescaledCols:
    Y_total.append(row[1])
    # split data in label and features
    row = numpy.delete(row, 1)
    X_total.append(row)

# X_train, X_test, Y_train, Y_test = train_test_split(X_total, Y_total, test_size=0.25)

# reg = LinearRegression().fit(X_train, Y_train)
# pred = reg.predict(X_test)
# print("Linear regression train score: ", reg.score(X_train, Y_train))
# print("Linear regression test score: ", reg.score(X_test, Y_test))
# print("------------------------------------------------------")

# reg1 = Ridge(alpha=1).fit(X_train, Y_train)
# print("Ridge regression train score: ", reg1.score(X_train, Y_train))
# print("Ridge regression test score: ", reg1.score(X_test, Y_test))
# print("------------------------------------------------------")

# reg2 = linear_model.Lasso(alpha=0.0001).fit(X_train, Y_train)
# print("Lasso regression train score: ", reg2.score(X_train, Y_train))
# print("Lasso regression test score: ", reg2.score(X_test, Y_test))
# print("------------------------------------------------------")

# reg3 = KNeighborsRegressor(n_neighbors=7, weights="distance").fit(X_train, Y_train)
# print("KNN regression train score: ", reg3.score(X_train, Y_train))
# print("KNN regression test score: ", reg3.score(X_test, Y_test))
# print("------------------------------------------------------")
#
# poly = PolynomialFeatures(degree=3)
# X_test_poly = poly.fit_transform(X_test)
# X_train_poly = poly.fit_transform(X_train)
# lin_reg_poly = LinearRegression().fit(X_train_poly, Y_train)
# ridge_poly = Ridge(alpha=1).fit(X_train_poly, Y_train)
# knn_poly = reg3 = KNeighborsRegressor(n_neighbors=7, weights="distance").fit(X_train_poly, Y_train)
# print("Linear regression poly features train score: ", lin_reg_poly.score(X_train_poly, Y_train))
# print("Linear regression poly test score: ", lin_reg_poly.score(X_test_poly, Y_test))
# print("------------------------------------------------------")
# print("Ridge regression poly train score: ", ridge_poly.score(X_train_poly, Y_train))
# print("Ridge regression poly test score: ", ridge_poly.score(X_test_poly, Y_test))
# print("------------------------------------------------------")
# print("KNN regression poly train score: ", reg3.score(X_train_poly, Y_train))
# print("KNN regression poly test score: ", reg3.score(X_test_poly, Y_test))
# print("------------------------------------------------------")

# el_net_reg = ElasticNet(random_state=0)
# el_net_reg.fit(X_train, Y_train)
# print("Elastic-Net regression train score: ", el_net_reg.score(X_train, Y_train))
# print("Elastic-Net regression test score: ", el_net_reg.score(X_test, Y_test))
# print("------------------------------------------------------")
# el_net_reg.fit(X_train_poly, Y_train)
# print("Elastic-Net regression poly train score: ", el_net_reg.score(X_train_poly, Y_train))
# print("Elastic-Net regression poly test score: ", el_net_reg.score(X_test_poly, Y_test))
# print("------------------------------------------------------")


# the best score was obtained with this model
# X_new has less features, the least important ones are cut off
X_new = SelectKBest(f_regression, k=9).fit_transform(X_total, Y_total)
X_train_new, X_test_new, Y_train_new, Y_test_new = train_test_split(X_new, Y_total, test_size=0.25)
reg10best = KNeighborsRegressor(n_neighbors=7, weights="distance").fit(X_train_new, Y_train_new)

print("KNN regression train score with 10 best features: ", reg10best.score(X_train_new, Y_train_new))
print("KNN regression test score with 10 best features: ", reg10best.score(X_test_new, Y_test_new))

Y_predictions = reg10best.predict(X_test_new)
x_ticks = numpy.arange(1., len(Y_predictions) + 1, 1)

errors = []
for a, b in zip(Y_predictions, Y_test_new):
    errors.append(abs(a - b))

# plot true values versus predicted values
predictions = mpatches.Patch(color="green", label="Predictions")
actual_data = mpatches.Patch(color="cyan", label="Actual data")
plt.plot(x_ticks, Y_predictions, "g",  x_ticks, Y_test_new, "c")
plt.legend(handles=[predictions, actual_data])
plt.title("Predictions compared to actual results")
plt.show()

# plot error for each point
plt.plot(x_ticks, errors, "g")
plt.title("Error of prediction")
plt.show()


