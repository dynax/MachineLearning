
from sklearn import linear_model
import time
print("lr model")
x_train_flat = x_train.reshape([-1, 28*28])
x_test_flat = x_test.reshape([-1, 28*28])
lr_model = linear_model.LogisticRegression()
time_start = time.time()
lr_model.fit(x_train_flat, y_train)
time_end = time.time()
print("Elapsed time: %.2f" % (time_end - time_start))
lr_model.score(x_test_flat, y_test)
