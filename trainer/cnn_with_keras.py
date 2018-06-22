import keras as keras
from keras.models import Model
from keras.layers import Input, Dense, Activation,Flatten, Conv2D
from keras.layers import Dropout, MaxPooling2D
import keras.backend as K
from keras import callbacks
import tensorflow as tf
K.set_image_data_format('channels_last')
import numpy as np
import argparse
from tensorflow.python.lib.io import file_io

def model(input_shape):
##First input
    X_input = Input(input_shape)

    ##Convolutional Layer 1
    X = Conv2D(
    filters=32,
    kernel_size=(5, 5),
    strides=(1, 1),
    padding='same',
    name = 'conv1'
    )(X_input)
    X = Activation('relu')(X)

    ##Max pooling layer 1
    X = MaxPooling2D(pool_size=(2, 2), strides =2, name = 'maxpool1')(X)

    ##Convolutional Layer 2
    X = Conv2D(
    filters=64,
    kernel_size=[5,5],
    padding='same',
    name = 'conv2'
    )(X)
    X = Activation('relu')(X)

    ##Max Pooling Layer 2
    X = MaxPooling2D(pool_size=(2, 2), strides =2, name = 'maxpool2')(X)

    ##Flatten
    X = Flatten()(X)

    ##Dense Layer
    X = Dense(1024, activation='relu', name='dense_1')(X)

    ##Dropout layer
    X = Dropout(0.4, name = 'dropout')(X)

    ##dense 2 layer
    X = Dense(10, activation='softmax', name ='dense_2')(X)

    ##The model object
    model = Model(inputs = X_input, outputs = X, name='cnnMINSTModel')

    return model



def main(job_dir,**args):

    ##Setting up the path for saving logs
    logs_path = job_dir + 'logs/tensorboard'

    ##Using the GPU
    with tf.device('/device:GPU:0'):

        ##Loading the data
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        train_data = mnist.train.images  # Returns np.array
        train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        eval_data = mnist.test.images  # Returns np.array
        eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

        ##Pre processing the data
        train_labels = keras.utils.np_utils.to_categorical(train_labels, 10)
        eval_labels = keras.utils.np_utils.to_categorical(eval_labels, 10)
        train_data = np.reshape(train_data, [-1, 28, 28, 1])
        eval_data = np.reshape(eval_data, [-1,28,28,1])

        ## Initializing the model
        Model = model(train_data.shape[1:]);

        ## Compling the model
        Model.compile(optimizer = "Adam" , loss = "binary_crossentropy", metrics = ["accuracy"]);

        ## Printing the modle summary
        Model.summary()

        ## Adding the callback for TensorBoard and History
        tensorboard = callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)

        ##fitting the model
        Model.fit(x = train_data, y = train_labels, epochs = 4,verbose = 1, batch_size=100, callbacks=[tensorboard], validation_data=(eval_data,eval_labels) )

        # Save model.h5 on to google storage
        Model.save('model.h5')
        with file_io.FileIO('model.h5', mode='r') as input_f:
            with file_io.FileIO(job_dir + 'model/model.h5', mode='w+') as output_f:
                output_f.write(input_f.read())


##Running the app
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)
