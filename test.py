import numpy as np
import matplotlib.pyplot as plt
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' ##for avx command set in CPU :https://blog.csdn.net/hq86937375/article/details/79696023
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras import optimizers
from keras.layers.normalization import BatchNormalization #ref:https://www.zhihu.com/question/55621104
from keras import initializers
from keras import callbacks

np.random.seed(10)

#interface variable
name=""  

#user interface
def main():
    if len(sys.argv) < 3: # 
        print("Usage:", sys.argv[0], "--name <test name>")
        sys.exit(1)       # 
    if sys.argv[1] != '--name': # 
        print("Usage:", sys.argv[0], "--name <test name>")
        sys.exit(1)       #
    global name
    name = sys.argv[2]
    

#declare variables
data = np.loadtxt('data.txt',dtype='float64')
def get_data():
    return data
training = {}
training['Rate'] = 0.1 
training['Region'] = 0.0
training['Validation_split'] = 0.2
training['Epochs'] = 300
training['Data'] = data[int(np.floor(data.shape[0]*training['Region'])) : int(np.floor(data.shape[0]*training['Region']) + np.floor(data.shape[0]*training['Rate']))] 
training['BatchRate'] = 0.1
training['BatchSize'] = int(np.floor(training['Data'].shape[0]))
def show_train_history(training_history ,training_history_his_type1 = 'std_auc' ,training_history_his_type2 ='val_std_auc'):
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

inputLen = 20
input_array = training['Data'][:,:(inputLen)]
if __name__ == "__main__": 
    print(input_array)

output_index_array = [ (65-1) , (21-1)] # 65.relic 67.DM mass
raw_output_array = training['Data'][:, output_index_array]
if __name__ == "__main__":
    print(raw_output_array)

#rescaling
output_layer_std = np.std(raw_output_array,axis=0)
output_layer_mean = np.mean(raw_output_array,axis=0)
if __name__ == "__main__":
    print("output_mean:",end='')
    print(output_layer_mean)
    print("output_std:",end='')
    print(output_layer_std)
output_array = ( raw_output_array - np.array([output_layer_mean]).repeat( raw_output_array.shape[0] ,axis=0) ) \
               / (np.array([ output_layer_std ]).repeat( raw_output_array.shape[0] ,axis=0) )
if __name__ == "__main__":
    print("output_array:",end='')
    print(output_array)

def output_scale_recover(array):
    return ( array * np.array([output_layer_std]).repeat( array.shape[0] ,axis=0) ) \
               + (np.array([ output_layer_mean ]).repeat( array.shape[0] ,axis=0) )

#model
if __name__ == "__main__":
    main()
    model = Sequential()
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

    #metric function
    import keras.backend as K
    def std_auc(y_true, y_pred):
        #輸出誤差幾個標準差    
        return K.std((y_pred-y_true)/K.constant(output_layer_std))

    #loss function(desserted)
    def std_loss(y_true, y_pred):
        return K.var((y_pred-y_true)/K.constant(output_layer_std))

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

    #show and save result
    training['std_prediction'] =  model.predict(data[0:5,:(inputLen)])
    print(training['prediction']) 
    training['prediction'] =  output_scale_recover( model.predict(data[0:5,:(inputLen)]) )
    print(training['prediction'])

    # serialize model to JSON
    model_json = model.to_json()
    with open(name+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights( name+".h5")
    print("Saved model to disk")

    show_train_history(training['History'],'mean_absolute_error','val_mean_absolute_error')

    def Plot(y_array,z_array):
        plt.plot(y_array[:,0],y_array[:,1],"ko")
        plt.plot(z_array[:,0],z_array[:,1],"ro")
        plt.title('{relic to m_h}')
        plt.xlabel('relic')
        plt.ylabel('m_h')    
        plt.legend(['theory','model'],loc = 'upper left')
        plt.show()
        
    Plot(output_array,model.predict(input_array))
