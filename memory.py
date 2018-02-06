# read_dataset.py
# Author : Hiroki
# Modified: 2018-1-10
import cPickle
import matplotlib.pyplot as pl
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def load_dataset():
    data = unpickle('cifar-10/data_batch_1')
    images = [data['data'][i].reshape(3, 32, 32) for i in range(data['data'].shape[0])]
    pl.figure(figsize=[1.0, 1.0])
    pl.imshow(np.transpose(images[5], [1, 2, 0]))
    pl.show()
    return images

######## can we build a network to detect if the pixel belongs
######## to a edge according to its neighbors
# Each of the pixel in the RGB image will be turned into
# a vector of Edge-element{boolean: True means edge otherwise
# in-area pixel}:
class SparseMatrix(object):
    '''SparseMatrix: A vector of elements (v,[i,j,...])
    in which {v} is required to be nonzero and {[i,j,...]}
    is the coordination of this element in matrix.
    '''

    def __init__(self, elements=None, coords=None, dims=None):
        if elements!=None:
            assert type(elements)==np.ndarray
            self.elements = elements.astype(np.float32)
        if coords!=None:
            assert type(coords)==np.ndarray
            self.coords = coords.astype(np.int32)
        if dims!=None:
            self.dims = dims

    def __from_dense_matrix__(self, dense_matrix):
        '''dense_matrix required to be a numpy array'''
        assert (type(dense_matrix) == np.ndarray)
        self.elements = []
        self.coords = []
        # One dimension vector
        if len(dense_matrix.shape)==1:
            for i in range(dense_matrix.shape[0]):
                if dense_matrix[i]:
                    self.elements.append(dense_matrix[i])
                    self.coords.append(i)
            self.elements = np.array(self.elements)
            self.coords = np.array(self.coords)
            self.dims = dense_matrix.shape
        elif len(dense_matrix.shape)==2:
            for i in range(dense_matrix.shape[0]):
                for j in range(dense_matrix.shape[1]):
                    if dense_matrix[i,j]:
                        self.elements.append(dense_matrix[i,j])
                        self.coords.append([i,j])
            self.elements = np.array(self.elements)
            self.coords = np.array(self.coords)
            self.dims = dense_matrix.shape
        elif len(dense_matrix.shape)==3:
            for i in range(dense_matrix.shape[0]):
                for j in range(dense_matrix.shape[1]):
                    for k in range(dense_matrix.shape[2]):
                        if dense_matrix[i,j,k]:
                            self.elements.append(dense_matrix[i,j,k])
                            self.coords.append([i,j,k])
            self.elements = np.array(self.elements)
            self.coords = np.array(self.coords)
            self.dims = dense_matrix.shape
        else:
            raise ValueError("Matrix to convert is not supported!")

    def __multiply__(self, sparse_matrix):
        assert type(sparse_matrix)==SparseMatrix
        if len(sparse_matrix.dims)==1:
            if sparse_matrix.dims[0]==1:
                self.elements *= sparse_matrix.elements[0]
        elif ...

def conv(x, w):
    '''Conv: A plain convolution operation
    input [x] should be sparse matrix.
    param [w] is a kernel
    '''


def Test_Class_SparseMatrix():
    sm = SparseMatrix()
    x = np.random.uniform(0,1,[32,32])
    x[x < 0.99] = 0
    sm.__from_dense_matrix__(x)
    print(sm.elements)
    print(sm.coords)

def main():
    print('============================')
    Test_Class_SparseMatrix()

if __name__ == '__main__':
    main()