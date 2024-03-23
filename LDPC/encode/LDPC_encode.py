from typing import Optional

from LDPC.BiArray import *
from MatrixConstructor.HMatrixConstructor import *


class Encoder:
    def __init__(self, matrixConstructor: Optional[HMatrixConstructor]):
        self.matrixConstructor = matrixConstructor
        self.Kbit = matrixConstructor.Kbit
        self.Nbit = matrixConstructor.Nbit
        self.Mbit = matrixConstructor.Mbit
        self.H = None

    def __repr__(self):
        return f'Base class: Encoder'

    def encode(self, x, method, isVal):
        pass


class QCLdpcEncoder(Encoder):
    """ QC-LDPC编码 """

    def __init__(self, qcMatrix: Optional[QCMatrix]):
        super().__init__(qcMatrix)
        self.z = qcMatrix.z
        self.mb = qcMatrix.mb
        self.nb = qcMatrix.nb
        self.kb = self.nb - self.mb
        self.H = BiArray(qcMatrix.make)

    def __repr__(self):
        return f'QC-LDPC编码器，校验矩阵构造器：{self.matrixConstructor.__repr__()}'

    def encode(self, x: Union[list, np.ndarray, BiArray],
               method: Optional[str] = 'Quasi Cyclic Bidiagonal Fast encode', isVal=False):
        """
        QC-LDPC编码函数统一接口
        :param x: 输入的原码 Union[list, np.ndarray, BiArray]
        :param method: 选择编码的方法
        :param isVal: 是否进行校验，确认编码是否出错
        :return: 编码code [n,1] BiArray
        """
        # TODO:
        if method == 'Quasi Cyclic Bidiagonal Fast encode':
            code = self.QuasiCyclicBidiagonal_Fastencode(x)
            if isVal:
                zero_s = BiArray(np.zeros(self.Mbit).reshape((-1,1)))
                assert (self.H @ code == zero_s).all(), '编码出错'
            return code
        else:
            assert False, f'没有"{method}"编码方法'

    def QuasiCyclicBidiagonal_Fastencode(self, x: Union[list, np.ndarray, BiArray]):
        """ IEEE802.16e标准下的fast encode方法 """
        if not isinstance(self.matrixConstructor, IEEE80106eQCMatrix):
            raise TypeError

        self.Kbit = self.matrixConstructor.Kbit
        assert self.Kbit == x.shape[0], f'输入的源码shape应该是(Kbit,),而不是{x.shape}'

        s = BiArray(x).reshape((-1, 1))  # 输入
        mb = self.mb
        kb = self.kb
        z = self.z
        NonZero = self.matrixConstructor.NonZero

        def Hb1_i_j(i, j):
            return self.H[i * z: (i + 1) * z, j * z: (j + 1) * z]

        def s_i(s, i):
            return s[i * z: (i + 1) * z]

        p = []
        # 计算p0
        Zh0 = self.H[0: z, kb * z: (kb + 1) * z]
        Zhr = self.H[NonZero * z: (NonZero + 1) * z, kb * z: (kb + 1) * z]
        add_inv = Zh0 + Zhr
        add_inv = add_inv + self.H[(mb - 1) * z: mb * z, kb * z: (kb + 1) * z]
        add_inv = ~add_inv
        sum_num = BiArray(np.zeros(z)).reshape((-1, 1))  # 形状 [z, 1]
        for i in range(mb):
            for j in range(kb):
                sum_num = sum_num + Hb1_i_j(i, j) @ s_i(s, j)  # 形状 [z, z] * [z, 1] = [z, 1]
        p0 = add_inv @ sum_num
        p.append(p0)

        # 计算P1
        sum2 = BiArray(np.zeros(z)).reshape((-1, 1))
        for j in range(kb):
            sum2 = sum2 + Hb1_i_j(0, j) @ s_i(s, j)
        p1 = sum2 + Zh0 @ p[0]
        p.append(p1)

        # 计算剩余的mb-2个pi
        for i in range(2, mb):
            sumi = BiArray(np.zeros(z)).reshape((-1, 1))
            if i == NonZero + 1:  # r+1处是特例
                sumi = sumi + Zhr @ p[0]
            for j in range(kb):
                sumi = sumi + Hb1_i_j(i - 1, j) @ s_i(s, j)
            pi = sumi + p[i - 1]
            p.append(pi)

        # 重新还原出码字
        c = BiArray(np.zeros(self.Nbit)).reshape((-1, 1))
        c[0:kb * z] = s
        for i in range(mb):
            c[kb * z + i * z: kb * z + (i + 1) * z] = p[i]
        c.reshape(-1)

        return c
