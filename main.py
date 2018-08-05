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
import data2 as data2
import model_train as model_train

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
data = data2.get_data()

training = {}
training['Rate'] = 0.1 
training['Region'] = 0.0
training['Validation_split'] = 0.2
training['Epochs'] = 300
training['Data'] = data[int(np.floor(data.shape[0]*training['Region'])) : int(np.floor(data.shape[0]*training['Region']) + np.floor(data.shape[0]*training['Rate']))] 
training['BatchRate'] = 0.1
training['BatchSize'] = int(np.floor(training['Data'].shape[0]))

##input
inputLen = 20
input_array = training['Data'][:,:(inputLen)]
input_mask = np.ones(inputLen, dtype=bool)
input_mask[[(1-1),(12-1),(13-1),(14-1),(17-1),(18-1),(19-1),(20-1)]] = False
input_array = input_array[:,input_mask]
inputLen = input_array.shape[1]

##output
output_index_array = [ (65-1) , (21-1)] # 65.relic 67.DM mass
raw_output_array = training['Data'][:, output_index_array]
#rescaling
output_layer_std = np.std(raw_output_array,axis=0)
output_layer_mean = np.mean(raw_output_array,axis=0)
output_array = ( raw_output_array - np.array([output_layer_mean]).repeat( raw_output_array.shape[0] ,axis=0) ) \
               / (np.array([ output_layer_std ]).repeat( raw_output_array.shape[0] ,axis=0) )

if __name__ == "__main__": 
    print("input_array:",end='')
    print(input_array)
    print("raw_output_array:",end='')
    print(raw_output_array)
    print("output_mean:",end='')
    print(output_layer_mean)
    print("output_std:",end='')
    print(output_layer_std)
    print("output_array:",end='')
    print(output_array)

##helper function
def output_scale_recover(array):
    return ( array * np.array([output_layer_std]).repeat( array.shape[0] ,axis=0) ) \
               + (np.array([ output_layer_mean ]).repeat( array.shape[0] ,axis=0) )
def Plot(y_array,z_array):
        plt.plot(y_array[:,0],y_array[:,1],"ko")
        plt.plot(z_array[:,0],z_array[:,1],"ro")
        plt.title('{relic to m_h}')
        plt.xlabel('relic')
        plt.ylabel('m_h')    
        plt.legend(['theory','model'],loc = 'upper left')
        plt.show()

#model
if __name__ == "__main__":
    main()
    model = model_train.load(name,input_array,output_array,training,data)
    #show and save result
    training['std_prediction'] =  model.predict(data[0:5,:(inputLen)])
    print(training['prediction']) 
    training['prediction'] =  output_scale_recover( model.predict(data[0:5,:(inputLen)]) )
    print(training['prediction'])
    Plot(output_array,model.predict(input_array))
