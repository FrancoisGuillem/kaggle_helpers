import numpy as np
import time
from typing import List
from matplotlib import pyplot as plt
import json

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

    sketches = keras.preprocessing.sequence.pad_sequences(sketches, max_len + 1, value=0, dtype="float32", padding="post")

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
        sketch[:, 2] = (sketch[:, 2] + 1) / 2
    lines = np.split(sketch, np.where(sketch[:,2] < 0.5)[0][1:], axis=0)
    plot_sketch(lines, title)
