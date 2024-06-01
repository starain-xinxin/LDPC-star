from typing import Optional

import numpy as np
from numpy import ndarray

import Chain.channel
from LDPC.BiArray import *
from MatrixConstructor.HMatrixConstructor import *

BF_Decode_MAX_Iter = 25
SPA_Decode_MAX_Iter = 25
WBF_Decode_MAX_Iter = 25
MSA_Decode_MAX_Iter = 25


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


# noinspection SpellCheckingInspection,DuplicatedCode
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

        elif method == 'LLR-BP' or method == 'BP':
            if max_iter is None:
                decode, flag = self.LLR_BP_decode(code, channel)
            else:
                decode, flag = self.LLR_BP_decode(code, channel, max_iter=max_iter)
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

        elif method == 'SPA':
            if max_iter is None:
                decode, flag = self.WBF_decode(code)
            else:
                decode, flag = self.WBF_decode(code, max_iter=max_iter)
            if (not flag) and display:
                print(f'译码失败：{self.__repr__()} 译码方法{method}')
            return decode.reshape(-1)[:self.Kbit]

        elif method == 'MSA':
            if max_iter is None:
                decode, flag = self.MSA_decode(code, channel)
            else:
                decode, flag = self.MSA_decode(code, channel, max_iter=max_iter)
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
    def LLR_BP_decode(self, code:np.ndarray, channel:Chain.channel.Channel, max_iter=SPA_Decode_MAX_Iter):
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

    def SPA_decode(self, code, channel, max_iter=SPA_Decode_MAX_Iter):
        """ SPA解码算法实现
        TODO: 兼容性问题，接口不完全统一，算法效率可以提升
        """
        H = self.H.numpy
        if isinstance(code, np.ndarray):
            received_signal = code
        else:
            assert False, f'SumProduct解码error：输入类型为{type(code)},应为numpy.ndarray'
        sigma = channel.std
        decoded = np.zeros(received_signal.shape)
        receivedP = self.pfunction(received_signal, sigma)
        receivedP = receivedP.reshape(-1, H.shape[1])

        flag = False
        for j in range(len(receivedP)):
            receivedPzero = receivedP[j]
            qNeg = np.multiply(receivedPzero, H)
            H_mask = H != 0
            k = np.zeros(qNeg.shape)
            for _ in range(max_iter):
                rPos, rNeg = self.check2bit(qNeg, H_mask)
                rowProductPos, qPostmp2 = self.bit2check(rPos, receivedPzero)
                rowProductNeg, qNegtmp2 = self.bit2check(rNeg, receivedPzero, neg=True)
                k[H_mask] = np.divide(1, qPostmp2[H_mask] + qNegtmp2[H_mask])
                qNeg = np.multiply(k, qNegtmp2)

            QtmpPos = np.multiply(1 - receivedPzero, rowProductPos)
            QtmpNeg = np.multiply(receivedPzero, rowProductNeg)
            K = np.divide(1, QtmpPos + QtmpNeg)
            QPos = np.multiply(K, QtmpPos)
            decoded[j * H.shape[1]:j * H.shape[1] + H.shape[1]] = QPos > 0.5
            if np.all((H @ decoded) % 2 == 0):
                flag = True

        return decoded, flag

    @staticmethod
    def pfunction(received, sigma=1):
        """ SPA辅助函数 """
        return 1 / (1 + np.exp(2 * received / sigma**2))

    @staticmethod
    def check2bit(qNeg, H_mask):
        """ SPA辅助函数 """
        shape = qNeg.shape
        rPos, rNeg = np.zeros(shape), np.zeros(shape)
        qTmp = qNeg[H_mask].reshape(shape[0], -1)
        qTmp = 1 - 2 * qTmp
        colProduct = np.product(qTmp, axis=1)
        rTmpPos = 0.5 + 0.5 * np.divide(np.full(qTmp.shape, colProduct.reshape(shape[0], 1)), qTmp)
        rTmpNeg = 1 - rTmpPos
        rPos[H_mask] = rTmpPos.flatten()
        rNeg[H_mask] = rTmpNeg.flatten()
        return rPos, rNeg

    @staticmethod
    def bit2check(rPos, receivedPzero, neg=False):
        """ SPA辅助函数 """
        rPosMask = rPos == 0
        rPos[rPosMask] = 1
        rowProductPos = np.product(rPos, axis=0)
        rPostmp = np.divide(np.full(rPos.shape, rowProductPos), rPos)
        rPos[rPosMask] = 0
        rPostmp[rPosMask] = 0
        if not neg:
            qPostmp2 = np.multiply(1 - receivedPzero, rPostmp)
        else:
            qPostmp2 = np.multiply(receivedPzero, rPostmp)
        return rowProductPos, qPostmp2

    def MSA_decode(self, received_signal, channel, max_iter=10, norm_factor=0.5):
        """ Min-Sum Algorithm
        TODO:实现的有问题
        """
        H = self.H.numpy
        N = H.shape[1]
        self.norm_factor = norm_factor
        self.HRowNum, self.HColNum = self.generate_indices(H)
        vl = channel.initialize_llr(received_signal)
        decoderData = np.zeros(N)

        uml = np.zeros(np.sum(self.HColNum))
        vml = np.zeros(np.sum(self.HColNum))

        col_start = 0
        for L in range(len(self.HColNum)):
            vml[col_start:col_start+self.HColNum[L]] = vl[L]
            col_start += self.HColNum[L]

        flag = False
        for iteration in range(max_iter):
            # Check nodes information process
            for L_r in range(len(self.HRowNum)):
                L_col = self.HRowNum[L_r]
                vmltemp = vml[L_col]
                vml_mark = np.ones(vmltemp.shape)
                vml_mark[vmltemp < 0] = -1
                vml_mark = np.prod(vml_mark)
                minvml = np.sort(np.abs(vmltemp))
                for L_col_i in range(len(L_col)):
                    if minvml[0] == abs(vmltemp[L_col_i]):
                        if vmltemp[L_col_i] < 0:
                            vmltemp[L_col_i] = -vml_mark * minvml[1]
                        else:
                            vmltemp[L_col_i] = vml_mark * minvml[1]
                    else:
                        if vmltemp[L_col_i] < 0:
                            vmltemp[L_col_i] = -vml_mark * minvml[0]
                        else:
                            vmltemp[L_col_i] = vml_mark * minvml[0]
                uml[L_col] = self.norm_factor * vmltemp

            # Variable nodes information process
            col_start = 0
            qn0_1 = np.ones(N)
            for L in range(len(self.HColNum)):
                umltemp = uml[col_start:col_start+self.HColNum[L]]
                temp = np.sum(umltemp)
                qn0_1[L] = temp + vl[L]
                umltemp = temp - umltemp
                vml[col_start:col_start+self.HColNum[L]] = umltemp + vl[L]
                col_start += self.HColNum[L]

            # Decision decoding
            decoderData[qn0_1 >= 0] = 0
            decoderData[qn0_1 < 0] = 1
            if np.all((H @ decoderData) % 2 == 0):
                flag = True
                break

        return decoderData, flag

    @staticmethod
    def generate_indices(H):
        r_mark, c_mark = np.where(H != 0)
        HColNum = np.sum(H, axis=0)
        HRowNum = [np.where(r_mark == row)[0] for row in range(H.shape[0])]
        return HRowNum, HColNum