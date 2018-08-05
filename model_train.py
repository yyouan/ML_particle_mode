import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' ##for avx command set in CPU :https://blog.csdn.net/hq86937375/article/details/79696023
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras import optimizers
from keras.layers.normalization import BatchNormalization #ref:https://www.zhihu.com/question/55621104
from keras import initializers
from keras import callbacks

def load( name,input_array,output_array,training,data ):
    model = Sequential()
    inputLen = input_array.shape[1]
    normal = initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)
    dense_1 = Dense(units = 10000 , input_dim = inputLen , kernel_initializer=normal,
                    bias_initializer=normal , activation = 'relu')
    DROP = Dropout(0.9)
    BN_1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
    dense_2 = Dense(units = 100 , input_dim = inputLen , kernel_initializer=normal,
                    bias_initializer=normal , activation = 'relu')
    dense_3 = Dense(units = 100 , input_dim = inputLen , kernel_initializer=normal,
                    bias_initializer=normal , activation = 'relu')
    output_layer = Dense( units = output_array.shape[1] , kernel_initializer = normal , activation = 'linear') #only positive value

    layer_list = [dense_1 ,dense_2,dense_3,output_layer]   

    #main code:
    ## create model
    for layer in layer_list:
        model.add(layer)
    print(model.summary())
    training['prediction'] =  model.predict(data[-5:-1,:(inputLen)]) 
    print(training['prediction'])
    ##compile model
    adam = optimizers.Adam(lr=0.001, beta_1=0.5, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False)
    model.compile(optimizer=adam,
                loss= 'mse',
                metrics=['mae'])
    ##time evalution:
    import time
    start_time = time.time()
    ##training mode
    training['Callbacks'] = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    filepath = name + "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    training['Checkpoint'] = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    training['History'] = model.fit(input_array,output_array
                                ,validation_split = training['Validation_split']
                                ,epochs = training['Epochs']
                                ,batch_size=training['BatchSize'] ,verbose=1 , callbacks = []) #verbose for show training process

    print("--- %s seconds ---" % (time.time() - start_time))

    #overall test:(desserted)
    '''
    training['Overall_accuracy'] = model.evaluate(data[:,:(inputLen)] ,data[:, output_index_array])
    print()
    print('Overall_accuracy',training['Overall_accuracy'][1])
    '''

    # serialize model to JSON
    model_json = model.to_json()
    with open(name+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights( name+".h5")
    print("Saved model to disk")

    show_train_history(name,training['History'],'mean_absolute_error','val_mean_absolute_error')
 
    return model

def show_train_history(name,training_history ,training_history_his_type1 = 'std_auc' ,training_history_his_type2 ='val_std_auc'):
        
        plt.subplot(211)
        plt.plot(training_history.history[training_history_his_type1])
        plt.plot(training_history.history[training_history_his_type2])
        plt.title('Traing History(unit:epoch)')
        plt.ylabel('acc(unit:std)')    
        plt.legend(['train','validation'],loc = 'upper left')

        plt.subplot(212)
        plt.plot(training_history.history['loss'])
        plt.plot(training_history.history['val_loss'])
        plt.title('Traing History(unit:epoch)')
        plt.ylabel('loss')
        plt.legend(['loss','val_loss'],loc = 'upper left')

        plt.savefig(name)
        plt.show()

