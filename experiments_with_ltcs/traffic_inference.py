# pip install -r reqTraffic.txt

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"# Run on CPU

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import pandas as pd
import datetime as dt

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()




def load_trace():
    df = pd.read_csv("data/traffic/Metro_Interstate_Traffic_Volume.csv")
    holiday = (df["holiday"].values == None).astype(np.float32)
    temp = df["temp"].values.astype(np.float32)
    temp -= np.mean(temp)  # normalize temp by annual mean
    rain = df["rain_1h"].values.astype(np.float32)
    snow = df["snow_1h"].values.astype(np.float32)
    clouds = df["clouds_all"].values.astype(np.float32)
    date_time = df["date_time"].values
    # 2012-10-02 13:00:00
    date_time = [dt.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in date_time]
    weekday = np.array([d.weekday() for d in date_time]).astype(np.float32)
    noon = np.array([d.hour for d in date_time]).astype(np.float32)
    noon = np.sin(noon * np.pi / 24)

    features = np.stack([holiday, temp, rain, snow, clouds, weekday, noon], axis=-1)

    traffic_volume = df["traffic_volume"].values.astype(np.float32)
    traffic_volume -= np.mean(traffic_volume)  # normalize
    traffic_volume /= np.std(traffic_volume)  # normalize

    return features, traffic_volume


def cut_in_sequences(x, y, seq_len, inc=1):

    sequences_x = []
    sequences_y = []

    for s in range(0, x.shape[0] - seq_len, inc):
        start = s
        end = start + seq_len
        sequences_x.append(x[start:end])
        sequences_y.append(y[start:end])

    return np.stack(sequences_x, axis=1), np.stack(sequences_y, axis=1)


class TrafficData:
    def __init__(self, seq_len=32):

        x, y = load_trace()

        train_x, train_y = cut_in_sequences(x, y, seq_len, inc=4)
        # just 100 training examples
        #train_x, train_y = cut_in_sequences(x[:100,:], y[:100], seq_len, inc=4)

        self.train_x = np.stack(train_x, axis=0)
        self.train_y = np.stack(train_y, axis=0)
        total_seqs = self.train_x.shape[1]
        print("Total number of training sequences: {}".format(total_seqs))
        permutation = np.random.RandomState(23489).permutation(total_seqs)
        valid_size = int(0.1 * total_seqs)
        test_size = int(0.15 * total_seqs)

        self.valid_x = self.train_x[:, permutation[:valid_size]]
        self.valid_y = self.train_y[:, permutation[:valid_size]]
        self.test_x = self.train_x[:, permutation[valid_size : valid_size + test_size]]
        self.test_y = self.train_y[:, permutation[valid_size : valid_size + test_size]]
        self.train_x = self.train_x[:, permutation[valid_size + test_size :]]
        self.train_y = self.train_y[:, permutation[valid_size + test_size :]]


if __name__ == "__main__":

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./tf_sessions/traffic/lstm.meta')
        saver.restore(sess, "./tf_sessions/traffic/lstm")


        graph = tf.get_default_graph()


        data = TrafficData()
        dp = data.train_x[:,1,:].reshape(32,1,7)
        print('inference features ',dp.shape)
        print('x-placeholder ', graph.get_operation_by_name("Placeholder").outputs[0])

        dpo = np.zeros(32).reshape(32,1)
        print('non-values ',dpo.shape)
        print('y-placeholder', graph.get_operation_by_name("Placeholder_1").outputs[0])

        y_value = 'dense/bias:0'

        z = graph.get_operation_by_name("Placeholder").outputs[0]
        c = graph.get_operation_by_name("Placeholder_1").outputs[0]

        # {'dense/bias/Adam:0', 'dense/bias/Adam_1:0', 'dense/bias:0'}
        out = sess.run(graph.get_tensor_by_name(y_value), feed_dict={z: dp})
        print('prediction/inference ',out)