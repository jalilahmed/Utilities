import numpy as np
from keras.utils import Sequence, to_categorical
from sklearn.utils import shuffle


class DataGenerator(Sequence):
    def __init__(self, data, labels, batch_size=32, dim=(32, 32, 32), n_channels=1, n_classes=1, shuffle=False):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.data = data
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        indexes = [index * self.batch_size, (index + 1) * self.batch_size]
        self.x_batched, self.y_batched = self.__data_generation(indexes)
        return self.x_batched, self.y_batched

    def on_epoch_end(self):
        if self.shuffle:
            shuffle(self.x_batched, self.y_batched)

    def __data_generation(self, indexes):
        start_index = indexes[0]
        end_index = indexes[1]
        x_batched = self.data[start_index:end_index]
        y_batched = self.labels[start_index:end_index]

        return x_batched, to_categorical(y_batched, num_classes=self.n_classes)

