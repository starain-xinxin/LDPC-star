import numpy as np
from typing import Optional
from MatrixConstructor.HMatrixConstructor import *
from LDPC.BiArray import *

BF_Decode_MAX_Iter = 50


class Decoder:

    def __init__(self, matrixConstructor: Optional[HMatrixConstructor]):
        # TODO
        self.matrixConstructor = matrixConstructor
        self.Kbit = matrixConstructor.Kbit
        self.Nbit = matrixConstructor.Nbit
        self.Mbit = matrixConstructor.Mbit
        self.H = None
        pass

    def __repr__(self):
        return f'Base class: Decoder'

    def decode(self, code):
        pass

class LdpcDecoder(Decoder):
    """ LDPC解码类的基类 """
    def __init__(self, LdpcMatrix:Union[QCMatrix]):
        super().__init__(LdpcMatrix)
        self.H = BiArray(LdpcMatrix.make)

    def __repr__(self):
        return f'LDPC朴素解码器, 校验矩阵构造器：{self.matrixConstructor.__repr__()}'

    def decode(self, code):
        pass

    def BF_decode(self, code:Union[list, np.ndarray, BiArray], maxiter=BF_Decode_MAX_Iter):
        """
        基础的比特翻转算法
        :param maxiter: 最大迭代次数
        :param code: 接受一维(n,)或者(1, n)
        :returns
        """
        code = BiArray(code).reshape((-1, 1))  # 输入[n,1],BiArray
        zero_s = BiArray(np.zeros(self.Mbit).reshape((-1,1)))
        for i in range(maxiter):
            s = self.H @ code   # 计算校验向量[m,1], BiArray
            if (s == zero_s).all():
                return code, True

            fn = (self.H.T.numpy @ s.numpy)   # 错误统计向量[n,1], ndarray
            max_value = np.max(fn)
            max_indices = np.where(fn == max_value)[0]
            chosen_index = np.random.choice(max_indices)

            fn_max = np.zeros_like(fn)
            fn_max[chosen_index] = 1

            fn_max = BiArray(fn_max)        # 比特翻转向量[n,1], BiArray

            code = code + fn_max   # 比特翻转
        return code, False





