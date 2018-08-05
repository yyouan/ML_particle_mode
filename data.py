import numpy as np

class Data:

    def load(self,filename):
        self.data = np.loadtxt(filename,dtype='float64')
        
    def __init__(self,data):
        self.data = data
        self.is_normalize = False
        self.std = 0
        self.mean = 0

    def remove_index(self,index_array):
    
    def front_index_lencut(self,inputLen):
        return self.data[:,:(inputLen)]
    
    def indexcut(self,index_array):
         return self.data[:, index_array]
    
    def len_ratecut(self,region,rate):
         return self.data[int(np.floor(self.data.shape[0]*region)) : int(np.floor(self.data.shape[0]*region) + np.floor(self.data.shape[0]*rate))]
    
    def normalize(self):
        self.std = np.std(self.data,axis=0)
        self.mean = np.mean(self.data , axis=0)
        self.data = ( self.data - np.array([self.mean]).repeat( self.data.shape[0] ,axis=0) ) \
               / (np.array([ self.std ]).repeat( self.data.shape[0] ,axis=0) )

if(__name__ == '__main__'):
    testdata = Data(np.array([]))
    testdata.load('data.txt')
    print(testdata.data)
    testdata2 = Data(testdata.front_index_lencut(20))
    testdata2.normalize()
    print(testdata2.std , testdata2.mean ,testdata2.data)
