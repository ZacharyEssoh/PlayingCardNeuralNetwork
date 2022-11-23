
## Playing Card Neural Network

In this project, I needed to create a convolutional Neural Network to classify images of a chosen dataset into catagories using Tensorflow and Keras libraries 


## Authors

Zachary Essoh


## Library References


```http
  Tensorflow
```
- This library allows for fast numerical computations for tasks. This library helps with setting the parameters of training models and running the epoch traing steps efficiently.
```http
  Keras
```
- This library allows for fast development of models for various high-level tasks like binary image classification and facial recognition. 


## Deployment

The user also needs to ensure that the dataset files are in the same file as the program, so that the program can properly import it and use it.

To deploy this project one just needs to download the Python file and open it in their preffered python IDE and intall the pyhon dependancies below, then click the run button. 

To deploy this project I had to install and import the Tensorflow and Keras Libraries.

```bash
import glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
```

## Screenshots

In these testing results I used a batch size of 100 and an epoch step size of 25.

![Training Results](https://user-images.githubusercontent.com/78782929/203472168-5feaa126-4f56-4a36-b1f0-78cd304d4848.png)

These are the first 16 cards of the training dataset.

![16 Cards](https://user-images.githubusercontent.com/78782929/203472306-3a6ee97f-92fe-4e9c-8b06-95caf4a91f05.png)



## Acknowledgements

 [Tensorflow Image Classification Tutorial](https://www.tensorflow.org/tutorials/images/classification)
 
 [Playing Card Dataset Used](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification)



