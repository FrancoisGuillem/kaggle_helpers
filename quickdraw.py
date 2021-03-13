import numpy as np
import time
from typing import List
from matplotlib import pyplot as plt
import json
import tensorflow.keras as keras

def read_sketches(classes: List[str], examples_per_class: int)-> np.array:
    """
    Read quickdraw data
    """
    data = []
    start_time = time.time()

    for i, cl in enumerate(classes): #enumerate(classes):
        with open(basedir + cl + ".ndjson") as f:
            nrows = 0
            for row in f:
                row = json.loads(row)
                lines = []
                for line in row["drawing"]:
                    new_line = np.ones_like(line[0])
                    new_line[0] = 0
                    line = np.array([
                        line[0],
                        line[1],
                        new_line
                    ])
                    lines.append(line.transpose())

                data.append([lines, i])
                nrows = nrows + 1
                if nrows >= examples_per_class: break

    random.shuffle(data)
    data = np.array(data)
    print(f'{len(data)} sketches parsed in {int(time.time() - start_time)} seconds')
    return data


def plot_sketch(sketch, title=""):
    for l in sketch:
        plt.plot(l[:,0], -l[:,1])
    plt.axis("off")
    plt.axis('equal')
    plt.title(title)


def preprocess_sketches(sketches, shuffle=True, diff=True, max_len=None):

    for i, sketch in enumerate(sketches):
        sketch = np.copy(sketch)
        if shuffle:
            random.shuffle(sketch)

        sketch = np.vstack(sketch)

        if diff:
            sketch = np.vstack([np.zeros(3), sketch])
            sketch[1:, 0:2] = np.diff(sketch[:, 0:2], axis=0)
            sketch = sketch[1:]

        sketches[i] = sketch

    if max_len is None:
        max_len = np.max([x.shape[0] for x in sketches])

    sketches = keras.preprocessing.sequence.pad_sequences(sketches, max_len + (1 if diff else 0), value=0, dtype="float32", padding="post")

    # scale all variable so that their domain is [-1, 1]
    if diff:
        sketches[:,:,0:2] = sketches[:,:,0:2] / 255
    else:
        sketches[:,:,0:2] = sketches[:,:,0:2] / 127.5 - 1
    sketches[:,:,2] = sketches[:,:,2] * 2 - 1

    return sketches


def plot_sketch_processed(sketch, title="", diff=True):
    """
    plot a sketch
    """
    if diff:
        sketch = np.copy(sketch)
        sketch[:,0:2] = np.cumsum(sketch[:,0:2], axis=0)

    lines = np.split(sketch, np.where(sketch[:,2] < 0)[0][1:], axis=0)
    plot_sketch(lines, title)


class DataIterator(keras.utils.Sequence):
    def __init__(self, data, task_type, batch_size=64, max_len=None, shuffle_lines=False):
        self.data = data
        self.task_type = task_type
        self.batch_size = batch_size
        self.max_len = max_len
        self.shuffle_lines = shuffle_lines
        self.idx = np.arange(len(data))

    def __len__(self):
        return len(self.data)// self.batch_size

    def __getitem__(self, idx):
        batch = self.data[(idx*self.batch_size):((idx+1)*self.batch_size)]
        batch_X = preprocess_sketches([x[0] for x in batch], self.shuffle_lines, True, self.max_len)

        if self.task_type == "classification":
            batch_Y = np.array([x[1] for x in batch])
            return batch_X, batch_Y
        elif self.task_type == "ts":
            return batch_X[: ,:-1 , :], batch_X[: ,1: , :]
        elif self.task_type == "gen":
            return batch_X, batch_X

    def on_epoch_end(self):
        self.shuffle()

    def shuffle(self):
        np.random.shuffle(self.data)


def train_valid_split(data, task_type, batch_size=64, max_len=None, shuffle_lines=False, validation_split=0.2):
    split_idx = int(len(data) * validation_split)
    valid = DataIterator(data[np.arange(split_idx)], task_type, batch_size, max_len, shuffle_lines)
    train = DataIterator(data[np.arange(split_idx, len(data))], task_type, batch_size, max_len, shuffle_lines)
    return train, valid
