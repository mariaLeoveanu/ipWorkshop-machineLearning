import pandas
import time
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt


def convert_to_timestamp(x):
    """Convert date objects to integers"""
    date_obj = datetime.datetime.strptime(x, "%Y-%m-%d")
    return time.mktime(date_obj.timetuple())


dates_to_plot = []
for day in range(14, 15):
    for hour in range(0, 24):
        date = convert_to_timestamp(f'2020-08-{day}')
        item = [date, hour]
        dates_to_plot.append(item)
print(dates_to_plot)

dataFrame = pandas.read_csv("AEP_hourly.csv", nrows=60000)


Y_labels = [x / 11706.34 for index, x in dataFrame[["AEP_MW"]].iterrows()]
X_features = []

for index, row in dataFrame[["Datetime"]].iterrows():
    row = row.to_string().split()
    X_features.append([convert_to_timestamp(row[1]), int(row[2].split(":")[0])])

feature_scaler = MinMaxScaler()
label_scaler = MinMaxScaler()

feature_scaler.fit(X_features)
label_scaler.fit(Y_labels)

X_features = feature_scaler.transform(X_features)
dates_to_plot = feature_scaler.transform(dates_to_plot)
Y_labels = label_scaler.transform(Y_labels)

X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y_labels, test_size=0.2)

reg = KNeighborsRegressor(n_neighbors=5).fit(X_train, Y_train)
# TODO: cross validation
print(reg.score(X_test, Y_test))

aug_predictions = reg.predict(dates_to_plot)
aug_predictions = label_scaler.inverse_transform(aug_predictions)
plt.plot(aug_predictions)
plt.ylabel("kWh per household")
plt.xticks(range(0, 24))
plt.grid(True)
plt.xlabel("Hour of the day")
plt.show()








































