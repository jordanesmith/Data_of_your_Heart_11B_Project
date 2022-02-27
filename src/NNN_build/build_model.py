import tensorflow as tf


def convolutional_block(x, filter, rate, pool=False):
    
    # copy tensor to variable called x_skip
    x_skip = x
    if pool: x_skip = tf.keras.layers.MaxPool1D()(x_skip)
    
    x = tf.keras.layers.BatchNormalization()(x) #TODO check axis here
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(rate)(x)
    x = tf.keras.layers.Conv1D(filter, 15, padding='same', strides=(1))(x) # TODO change hyperparameters here
    if pool: x = tf.keras.layers.AveragePooling1D()(x)
    
    # add residue TODO potentially do this better with connections between multiple layers 
    x = tf.keras.layers.Add()([x, x_skip])     
    
    return x

def NNN(shape = (max_sample_length, 1), classes = 3):
    
    # Setup Input Layer
    x_input = tf.keras.layers.Input(shape)
    # x = tf.keras.layers.ZeroPadding1D()(x_input) # TODO check if this padding is neccessary
    x = x_input
    
    # Define initial filter size
    filter_size = 15
    dropout_rate = 0.01
    
    # Build 16 convolutional blocks
    for i in range(16):
        # The filter size will go on increasing by a factor of 2
        filter_size = 1
        x = convolutional_block(x, filter_size, dropout_rate, pool = (i%3 == 0)) # only pool one in every 3 layers
    
    # End Dense Network
    n_dense_parameters = 32
    x = tf.keras.layers.Flatten()(x) # TODO check if this is neccessary
    x = tf.keras.layers.Dense(n_dense_parameters, activation='relu')(x) # optimise over this n_dense_parameters variable
    x = tf.keras.layers.Dense(classes, activation='softmax')(x) 
    model = tf.keras.models.Model(inputs=x_input, outputs=x, name="NNNet")
    return model