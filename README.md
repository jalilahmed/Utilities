# Utilities
This Repository contains functions in python for day to day programming specially focusing in domain of Domain Learning. You can find the function in the python with respective name. In this ReadMe file, a Short description of every function, its inputs, outputs, and their type along with a short example are given. 

## Motivation
The motivation for this project was to develop a log for all the different utilities function I write over the span of different Deep Learning Courses and Projects. This will help me keep a log of different helper functions I have written as well as provide an easy to use library for application in future project. Also I wanted to help new to community members as well, so they dont have to search on helper functions and they can focus more on the Deep Learning task at hand. 

## Dependencies
You will need these python packages before hand to use provided functions
  1. Numpy
  2. Keras
  3. Tensorflow
  
## Important Note
I have tried my best to develop class based functions for easy use. Just import the concerned .py file. Further you can make __init__.py file as well in your project for easy import of class object. I have added a simple version of __init__.py file as well. 

# Available Class

## DataGenerator

### Motivation
I wrote this function for using a large data.npy file for training my keras model. __It is based upon the tutorial given by [Shervine Amid](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html).__ I just refactored the code to be useful for numpy arrays. 

### Code
The DataGenerator class implements Sequence class of keras.utils. These function are overwritten
  1. __len__()
  2. __getitem()
  3. on_epoch_end()

### Constructor Inputs
  1. data : an numpy array of dimension (No. of Samples, Dimension, No. of Channels)
  2. labels: target labels
  3. batch_size: batch_size of your model. Generator will generate data with number of samples equal to batch_size
  4. dim: dimensions of  data
  5. n_channels: e.g. 3 for RGB Images
  6. n_classes: for one hot encoding of target labels
  7. shuffle: Boolean if you want to suffle data while generating. Default: False

### Usage
```
import DataGenerator
data = np.load('./data.npy')
labels = np.load('./labels.npy)

params = {'dim': input_shape,
          'batch_size': batch_size,
          'n_classes': n_classes,
          'n_channels': n_channels,
          'shuffle': False}
          
generator = DataGenerator(data, labels, **params)
```

### Further Resources
1. [Shervine Amid](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html)
2. [Keras Sequence Class](https://keras.io/preprocessing/sequence/)

*Please Note that this is tutorial based Class. *

## Credits
I would like to give credit to [lazyProgrammer](https://lazyprogrammer.me) for developing a very comprehensive and understandable course for deep learning. Also [Azeem Bootwala](https://github.com/azeembootwala) for encouragement. Also, 
1. [Shervine Amid](https://stanford.edu/~shervine/)

## License
MIT :copyright: [Jalil Ahmed](https://www.linkedin.com/in/jalil-siddiqui/).

