import numpy as np
from keras.utils import Sequence, to_categorical


class DataGenerator(Sequence):
    def __init__(self, data, labels, batch_size=32, dim=(32, 32, 32), n_channels=1, n_classes=1, shuffle=False):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.data = data
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        x_batched, y_batched = self.__data_generation(indexes)

        return x_batched, y_batched

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        start_index = indexes[0]
        end_index = indexes[1]

        x = self.data[start_index:end_index]
        y = self.labels[start_index:end_index]

        return x, to_categorical(y, num_classes=self.n_classes)

