
import numpy
import pandas
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNet

dataFrame = pandas.read_csv("SeoulBikeData.csv", encoding="unicode_escape")
dataFrame["Seasons"] = dataFrame["Seasons"].astype("category")
dataFrame["Seasons"] = dataFrame["Seasons"].cat.codes
dataFrame["Holiday"] = dataFrame["Holiday"].astype("category")
dataFrame["Holiday"] = dataFrame["Holiday"].cat.codes
dataFrame["Date"] = dataFrame["Date"].astype("category")
dataFrame["Date"] = dataFrame["Date"].cat.codes
all_cols = []
for i in range(0, 8760):
    line = []
    if dataFrame["Functioning Day"][i] == "Yes":
        for column in dataFrame.columns[:13]:
            line.append(dataFrame[column][i])
        all_cols.append(line)

scaler = StandardScaler()
scaler.fit(all_cols)
rescaledCols = scaler.transform(all_cols)

Y_total = []
X_total = []
for row in rescaledCols:
    Y_total.append(row[1])
    row = numpy.delete(row, 1)
    X_total.append(row)

X_train, X_test, Y_train, Y_test = train_test_split(X_total, Y_total, test_size=0.25)

reg = LinearRegression().fit(X_train, Y_train)
pred = reg.predict(X_test)
print("Linear regression train score: ", reg.score(X_train, Y_train))
print("Linear regression test score: ", reg.score(X_test, Y_test))
print("------------------------------------------------------")

reg1 = Ridge(alpha=1).fit(X_train, Y_train)
print("Ridge regression train score: ", reg1.score(X_train, Y_train))
print("Ridge regression test score: ", reg1.score(X_test, Y_test))
print("------------------------------------------------------")

reg2 = linear_model.Lasso(alpha=0.0001).fit(X_train, Y_train)
print("Lasso regression train score: ", reg2.score(X_train, Y_train))
print("Lasso regression test score: ", reg2.score(X_test, Y_test))
print("------------------------------------------------------")

reg3 = KNeighborsRegressor(n_neighbors=7, weights="distance").fit(X_train, Y_train)
print("KNN regression train score: ", reg3.score(X_train, Y_train))
print("KNN regression test score: ", reg3.score(X_test, Y_test))
print("------------------------------------------------------")

poly = PolynomialFeatures(degree=3)
X_test_poly = poly.fit_transform(X_test)
X_train_poly = poly.fit_transform(X_train)
lin_reg_poly = LinearRegression().fit(X_train_poly, Y_train)
ridge_poly = Ridge(alpha=1).fit(X_train_poly, Y_train)
knn_poly = reg3 = KNeighborsRegressor(n_neighbors=7, weights="distance").fit(X_train_poly, Y_train)
print("Linear regression poly features train score: ", lin_reg_poly.score(X_train_poly, Y_train))
print("Linear regression poly test score: ", lin_reg_poly.score(X_test_poly, Y_test))
print("------------------------------------------------------")
print("Ridge regression poly train score: ", ridge_poly.score(X_train_poly, Y_train))
print("Ridge regression poly test score: ", ridge_poly.score(X_test_poly, Y_test))
print("------------------------------------------------------")
print("KNN regression poly train score: ", reg3.score(X_train_poly, Y_train))
print("KNN regression poly test score: ", reg3.score(X_test_poly, Y_test))
print("------------------------------------------------------")

el_net_reg = ElasticNet(random_state=0)
el_net_reg.fit(X_train, Y_train)
print("Elastic-Net regression train score: ", el_net_reg.score(X_train, Y_train))
print("Elastic-Net regression test score: ", el_net_reg.score(X_test, Y_test))
print("------------------------------------------------------")
el_net_reg.fit(X_train_poly, Y_train)
print("Elastic-Net regression poly train score: ", el_net_reg.score(X_train_poly, Y_train))
print("Elastic-Net regression poly test score: ", el_net_reg.score(X_test_poly, Y_test))
print("------------------------------------------------------")