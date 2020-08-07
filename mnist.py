import numpy
import pandas
from PIL import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.manifold import Isomap
from matplotlib import pyplot as plt

df_train = pandas.read_csv("mnist_train.csv", header=None)
df_test = pandas.read_csv("mnist_test.csv", header=None)

X_test = []
Y_test = []
print("Started processing test data...")
for idx, row in df_test.iterrows():
    Y_test.append(row[0])
    curr_features = [row[i] for i in range(1, 785)]
    X_test.append(curr_features)
    if idx % 1000 == 0:
        print("Currently at row ", idx)

X_train = []
Y_train = []
print("Started processing training data...")
for idx, row in df_train.iterrows():
    Y_train.append(row[0])
    curr_features = [row[i] for i in range(1, 785)]
    X_train.append(curr_features)
    if idx % 1000 == 0:
        print("Currently at row ", idx)

decision_clf = DecisionTreeClassifier(max_depth=20)
decision_clf.fit(X_train, Y_train)
Y_prediction = decision_clf.predict(X_test)
print("Classifier test accuracy: ", decision_clf.score(X_test, Y_test))


iso = Isomap(n_components=2)
projection = iso.fit_transform(X_test)

# plotting the true classification
plt.scatter(projection[:, 0], projection[:, 1], c=Y_test)
plt.show()

# plotting the predicted classificaion
plt.scatter(projection[:, 0], projection[:, 1], c=Y_prediction)
plt.show()




