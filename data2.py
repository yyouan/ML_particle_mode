import numpy as np

##main code
data = np.loadtxt('data.txt',dtype='float64')

##send data
def get_data():
    return data

##test code:
if(__name__ == '__main__'):
    print('data2')
