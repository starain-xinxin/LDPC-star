from MatrixConstructor.HMatrixConstructor import *

# 测试IEEE80106eQCMatrix
a = IEEE80106eQCMatrix(Nbit=576, rate=1/2)
H_matrix = a.make
print(a)
print(H_matrix.shape)