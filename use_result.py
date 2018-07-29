import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from test import output_scale_recover,get_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' ##for avx command set in CPU :https://blog.csdn.net/hq86937375/article/details/79696023
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras import optimizers
from keras.layers.normalization import BatchNormalization #ref:https://www.zhihu.com/question/55621104
from keras import initializers
from keras import callbacks
from keras.models import model_from_json

np.random.seed(10)
inputLen = 20
data = get_data()

#interface variable
name=""
from_data = 0 
to_data = 1

#private:

#user interface
def main():
    if len(sys.argv) < 7: # 
        print("Usage:", sys.argv[0], "--name <test name> --from <the index test_data from> --to <the index test_data to>")
        sys.exit(1)       # 
    if sys.argv[1] != '--name': # 
        print("Usage:", sys.argv[0], "--name <test name> --from <the index test_data from> --to <the index test_data to>")
        sys.exit(1)       #
   
    if sys.argv[3] != '--from': # 
        print("Usage:", sys.argv[0], "--name <test name> --from <the index test_data from> --to <the index test_data to>")
        sys.exit(1)
    
    if sys.argv[5] != '--to': # 
        print("Usage:", sys.argv[0], "--name <test name> --from <the index test_data from> --to <the index test_data to>")
        sys.exit(1)
    global name
    name = sys.argv[2]
    print(name)
    global from_data
    from_data = int (sys.argv[4])
    global to_data
    to_data = int (sys.argv[6])
    print(from_data," to ",to_data)
        

if __name__ == '__main__':
    main()
    
# load json and create model
json_file = open( name +'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights( name + ".h5")
print("Loaded model from disk")

##compile model
adam = optimizers.Adam(lr=0.0001, beta_1=0.5, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False)
loaded_model.compile(optimizer=adam,
              loss= 'mse',
              metrics=['mae'])

##time evalution:
import time
start_time = time.time()
std_prediction =  loaded_model.predict(data[from_data:to_data,:(inputLen)])
print(std_prediction) 
prediction =  output_scale_recover( loaded_model.predict(data[from_data:to_data,:(inputLen)]) )
print(prediction)

