from typing import Optional

import numpy as np
from numpy import ndarray

import Chain.channel
from LDPC.BiArray import *
from MatrixConstructor.HMatrixConstructor import *

BF_Decode_MAX_Iter = 20
SPA_Decode_MAX_Iter = 20
WBF_Decode_MAX_Iter = 20


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

    def decode(self, code, method):
        pass


class LdpcDecoder(Decoder):
    """ LDPC解码类的基类 """

    def __init__(self, LdpcMatrix: Optional[QCMatrix]):
        super().__init__(LdpcMatrix)
        self.H = BiArray(LdpcMatrix.make)

    def __repr__(self):
        return f'LDPC朴素解码器, 校验矩阵构造器：{self.matrixConstructor.__repr__()}'

    def decode(self, code, method: Optional[str], channel=None, display=False, max_iter=None):
        """
        Ldpc解码函数统一接口

        :param max_iter: 迭代译码最大迭代次数
        :param channel: 信道
        :param display: 译码失败是否显示到终端
        :param code：接收一维(n,)或者(1, n)或者(n, 1) Union[list, np.ndarray, BiArray]
        :param method：接收一个字符串用于选择解码方法
        :return decode：译码后的码字(k,) ndarray
        """
        if method == 'BF':
            if max_iter is None:
                decode, flag = self.BF_decode(code)
            else:
                decode, flag = self.BF_decode(code, maxiter=max_iter)
            if (not flag) and display:
                print(f'译码失败：{self.__repr__()} 译码方法{method}')
            return decode.reshape(-1)[:self.Kbit].numpy

        elif method == 'SPA' or method == 'LLR-BP' or method == 'BP':
            if max_iter is None:
                decode, flag = self.SPA_decode(code, channel)
            else:
                decode, flag = self.SPA_decode(code, channel, max_iter=max_iter)
            if (not flag) and display:
                print(f'译码失败：{self.__repr__()} 译码方法{method}')
            return decode.reshape(-1)[:self.Kbit]

        elif method == 'WBF':
            if max_iter is None:
                decode, flag = self.WBF_decode(code)
            else:
                decode, flag = self.WBF_decode(code, max_iter=max_iter)
            if (not flag) and display:
                print(f'译码失败：{self.__repr__()} 译码方法{method}')
            return decode.reshape(-1)[:self.Kbit]

        elif method == 'None':
            return code.reshape(-1)[:self.Kbit].numpy
        else:
            assert False, f'没有"{method}"译码方法'

    def BF_decode(self, code: Union[list, np.ndarray, BiArray], maxiter=BF_Decode_MAX_Iter):
        """
        基础的比特翻转算法
        :param maxiter: 最大迭代次数
        :param code: 接收一维(n,)或者(1, n)或者(n, 1)
        :return decode: 译码但没截取的码字(n,1)
        :return flag: 是否译码成功
        """
        code = BiArray(code).reshape((-1, 1))  # 输入[n,1],BiArray
        zero_s = BiArray(np.zeros(self.Mbit).reshape((-1, 1)))
        for i in range(maxiter):
            s = self.H @ code  # 计算校验向量[m,1], BiArray
            if (s == zero_s).all():
                return code, True

            fn = (self.H.T.numpy @ s.numpy)  # 错误统计向量[n,1], ndarray
            max_value = np.max(fn)
            max_indices = np.where(fn == max_value)[0]
            chosen_index = np.random.choice(max_indices)

            fn_max = np.zeros_like(fn)
            fn_max[chosen_index] = 1

            fn_max = BiArray(fn_max)  # 比特翻转向量[n,1], BiArray

            code = code + fn_max  # 比特翻转
        return code, False

    # noinspection SpellCheckingInspection
    def SPA_decode(self, code:np.ndarray, channel:Chain.channel.Channel, max_iter=SPA_Decode_MAX_Iter):
        """
        SPA / LLR-BP算法
        本算法实现上利用了信道的BPSK调制的信息，因此不普适
        TODO：信道信息去耦合
        """
        H = self.H.numpy
        if isinstance(code, np.ndarray):
            received_signal = code
        else:
            assert False, f'SPA解码error：输入类型为{type(code)},应为numpy.ndarray'

        # H 矩阵的行（校验节点）和列（变量节点）数量
        num_checks, num_vars = H.shape

        # 从接收信号初始化 LLR 值
        LLR_initial = channel.initialize_llr(received_signal)

        # 初始化从变量节点到校验节点的消息
        LLRQ = np.zeros((num_checks, num_vars))

        # 初始化从校验节点到变量节点的消息
        LLRR = np.zeros((num_checks, num_vars))

        # 使用接收信号 LLR 值初始化 LLRQ
        for i in range(num_checks):
            for j in range(num_vars):
                if H[i, j] == 1:
                    LLRQ[i, j] = LLR_initial[j]

        decoded_bits: ndarray = np.zeros(num_vars)
        flag = False
        for iteration in range(max_iter):
            # 校验节点到变量节点的消息更新
            for i in range(num_checks):
                for j in range(num_vars):
                    if H[i, j] == 1:
                        product = 1
                        for k in range(num_vars):
                            if k != j and H[i, k] == 1:
                                product *= np.tanh(LLRQ[i, k] / 2)
                        LLRR[i, j] = 2 * np.arctanh(product)

            # 变量节点到校验节点的消息更新
            for j in range(num_vars):
                for i in range(num_checks):
                    if H[i, j] == 1:
                        sum_llr = 0
                        for k in range(num_checks):
                            if k != i and H[k, j] == 1:
                                sum_llr += LLRR[k, j]
                        LLRQ[i, j] = LLR_initial[j] + sum_llr

            # 计算每个变量节点的后验 LLR
            LQ = np.zeros(num_vars)
            for j in range(num_vars):
                sum_llr = 0
                for i in range(num_checks):
                    if H[i, j] == 1:
                        sum_llr += LLRR[i, j]
                LQ[j] = LLR_initial[j] + sum_llr

            # 解码比特
            for j in range(num_vars):
                if LQ[j] < 0:
                    decoded_bits[j] = 1
                else:
                    decoded_bits[j] = 0

            # 检查解码的比特是否满足所有校验
            if np.all((H @ decoded_bits) % 2 == 0):
                flag = True
                break

        return decoded_bits, flag

    def WBF_decode(self, code:np.ndarray, max_iter=WBF_Decode_MAX_Iter):
        H = self.H.numpy
        if isinstance(code, np.ndarray):
            received_signal = code
        else:
            assert False, f'WBF解码error：输入类型为{type(code)},应为numpy.ndarray'
        X, Y = H.shape
        E = np.zeros(Y)
        w = np.zeros(X)

        # 初始化解码比特
        z = np.zeros(Y)
        for ii in range(Y):
            if received_signal[ii] < 0.0:
                z[ii] = 0
            else:
                z[ii] = 1

        # 计算初始 syndrome
        s = np.mod(z @ H.T, 2)

        # 计算初始权重
        for i1 in range(X):
            Q1 = np.where(H[i1, :] == 1)[0]
            Y1 = np.abs(received_signal[Q1])
            w[i1] = np.min(Y1)

        flag = 0
        for I in range(max_iter):
            for i in range(Y):
                Q = np.where(H[:, i] == 1)[0]
                e = np.zeros(len(Q))
                for x in range(len(Q)):
                    e[x] = (2 * s[Q[x]] - 1) * w[Q[x]]
                E[i] = np.sum(e)

            max_E_index = np.argmax(E)
            z[max_E_index] = (z[max_E_index] + 1) % 2

            s = np.mod(z @ H.T, 2)
            if np.sum(s) == 0:
                flag = True
                break

        return z, flag