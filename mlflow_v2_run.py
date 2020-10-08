import argparse

import keras
import tensorflow as tf

import cloudpickle
from functools import wraps
import time

import mlflow
import mlflow.keras

def elapsed_time(func):
    @wraps(func)
    def out(*args, **kwargs):
        init_time = time.time()
        func(*args, **kwargs)
        elapsed_time = time.time() - init_time
        print(f"Elapsed time of {func.__name__}: {elapsed_time:.4f}")
    return out

@elapsed_time
def print_hello():
    time.sleep(1)
    print("Hello")

@elapsed_time
def main_func():
    parser = argparse.ArgumentParser(description='Train a Keras')
    parser.add_argument('--batch-size', '-b', type=int, default=8)
    parser.add_argument('--epochs', '-e', type=int, default = 4)
    parser.add_argument('--learning-rate', '-l', type=float, default = 0.1)
    parser.add_argument('--num-hidden-units', '-n', type=int, default=512)
    parser.add_argument('--dropout', '-d', type=float, default=0.5)
    parser.add_argument('--momentum', '-m', type=float, default=0.85)

    args = parser.parse_args()

    mlflow.log_param('batch_size', args.batch_size)
    mlflow.log_param('epochs', args.epochs)
    mlflow.log_param('learning_rate', args.learning_rate)

    mnist = keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=x_train[0].shape),
        keras.layers.Dense(args.num_hidden_units, activation=tf.nn.relu),
        keras.layers.Dropout(args.dropout),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    optimizer = keras.optimizers.SGD(lr=args.learning_rate, 
    momentum=args.momentum, nesterov=True)

    class LogMetricsCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            mlflow.log_metric('training_loss', logs['loss'], epoch)
            mlflow.log_metric('training_accuracy', logs['accuracy'], epoch)


    model.compile(optimizer=optimizer, 
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    model.fit(x_train, y_train, 
                epochs=args.epochs, batch_size=args.batch_size,
                callbacks=[LogMetricsCallback()])

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    mlflow.log_metric('test_loss', test_loss)
    mlflow.log_metric('test_accuracy', test_acc)

    mlflow.keras.log_model(model, artifact_path='keras-model')


if __name__ == "__main__":
    main_func()