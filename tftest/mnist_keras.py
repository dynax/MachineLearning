import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print("cnn model")
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)


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
