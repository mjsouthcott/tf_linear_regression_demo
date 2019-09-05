import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(5)

def generate_dataset():
    x_batch = np.random.randint(low = 0, high = 50, size = 12)
    x_batch.sort()
    y_batch = np.random.randint(low = 25, high = 100, size = 12)
    y_batch.sort()
    return x_batch, y_batch

def linear_regression():
    x = tf.placeholder(tf.float32, shape=(None, ), name='x')
    y = tf.placeholder(tf.float32, shape=(None, ), name='y')

    with tf.variable_scope('lreg') as scope:
        w = tf.Variable(np.random.normal(), name='W')
        b = tf.Variable(np.random.normal(), name='b')

        y_pred = tf.add(tf.multiply(w, x), b)

        loss = tf.reduce_mean(tf.square(y_pred - y))

    return x, y, y_pred, loss

def run():
    x_batch, y_batch = generate_dataset()
    x, y, y_pred, loss = linear_regression()

    optimizer = tf.train.GradientDescentOptimizer(0.0001)
    train_op = optimizer.minimize(loss)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        feed_dict = {x: x_batch, y: y_batch}

        for i in range(100000):
            session.run(train_op, feed_dict)
            print(i, 'loss:', loss.eval(feed_dict))
            """
            y_pred_batch = session.run(y_pred, {x: x_batch})
            plt.scatter(x_batch, y_batch)
            plt.title('TLAV Annual Operating Cost vs Odometer Reading')
            plt.xlabel('Odometer Reading (10^3 km)')
            plt.ylabel('Annual Operating Cost (10^3 $)')
            plt.plot(x_batch, y_pred_batch, color='red')
            plt.show(block=False)
            plt.pause(0.5)
            plt.close('all')
            """
        print('Predicting')
        y_pred_batch = session.run(y_pred, {x: x_batch})

    plt.scatter(x_batch, y_batch)
    plt.title('TLAV Annual Operating Cost vs Odometer Reading')
    plt.xlabel('Odometer Reading (10^3 km)')
    plt.ylabel('Annual Operating Cost (10^3 $)')
    plt.plot(x_batch, y_pred_batch, color='red')
    plt.savefig('plot.png')

if __name__ == "__main__":
    run()
