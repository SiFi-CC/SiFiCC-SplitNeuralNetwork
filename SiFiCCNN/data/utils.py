import numpy as np


def batch_generator(data, batch_size=32, epochs=None, shuffle=True):
    """
    Iterates over the data for the given number of epochs, yielding batches of
    size `batch_size`.
    :param data: np.array or list of np.arrays with the same first dimension;
    :param batch_size: number of samples in a batch;
    :param epochs: number of times to iterate over the data (default None, iterates
    indefinitely);
    :param shuffle: whether to shuffle the data at the beginning of each epoch
    :return: batches of size `batch_size`.
    """
    # TODO: UPDATE DOC STRING: SOURCE: SPEKTRAL PACKAGE
    if not isinstance(data, (list, tuple)):
        data = [data]
    if len(data) < 1:
        raise ValueError("data cannot be empty")
    if len({len(item) for item in data}) > 1:
        raise ValueError("All inputs must have the same __len__")

    if epochs is None or epochs == -1:
        epochs = np.inf
    len_data = len(data[0])
    batches_per_epoch = int(np.ceil(len_data / batch_size))
    epoch = 0
    while epoch < epochs:
        epoch += 1
        if shuffle:
            shuffle_inplace(*data)
        for batch in range(batches_per_epoch):
            start = batch * batch_size
            stop = min(start + batch_size, len_data)
            to_yield = [item[start:stop] for item in data]
            if len(data) == 1:
                to_yield = to_yield[0]

            yield to_yield


def shuffle_inplace(*args):
    rng_state = np.random.get_state()
    for a in args:
        np.random.set_state(rng_state)
        np.random.shuffle(a)
