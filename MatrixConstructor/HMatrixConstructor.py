import numpy as np
from typing import Union
import matplotlib.pyplot as plt
import seaborn as sns


class HMatrixConstructor:
    r"""构造校验矩阵的基类class
    - 所有的构造方式需要从此类继承
    """
    def __init__(self):
        self.Kbit = None
        self.Nbit = None
        self.Mbit = None
        self.H = None

    def __repr__(self):
        return f'Base class: VerificationMatrixConstructor'

    def make(self):
        pass


class QCMatrix(HMatrixConstructor):
    r"""朴素的QC-LDPC校验矩阵生成器
    Args:
        - Nbit：码长
        - Kbit：源消息长
        - Hb：基矩阵
    """
    def __init__(self, Nbit, Kbit, Hb: Union[list, np.ndarray], device='cpu'):
        super().__init__()
        self.Nbit = int(Nbit)
        self.Mbit = int(Nbit-Kbit)   # 校验长度
        self.Kbit = int(Kbit)   # 源消息长

        self.Hb = Hb
        if type(self.Hb) == list:
            self.Hb = np.array(self.Hb)

        self.mb, self.nb = self.Hb.shape
        self.z = int(self.Nbit / self.nb)
        assert self.Nbit % self.nb == 0 and self.Mbit % self.mb == 0 and self.z == self.Mbit / self.mb, '基矩阵与校验矩阵形状不匹配'

        self.device = device

        self.H = None

    def __repr__(self):
        return f'朴素的QC-LDPC校验矩阵生成器 [码长：{self.Nbit}, 源消息长：{self.Kbit} ]'

    @property
    def make(self):
        qc_matrix = np.zeros((self.Mbit, self.Nbit), dtype=int)
        for i in range(self.mb):
            for j in range(self.nb):
                qij = self.Hb[i, j]
                if qij == -1:
                    expandMatrix = np.zeros((self.z, self.z), dtype=int)
                elif qij == 0:
                    expandMatrix = np.eye(self.z, dtype=int)
                else:
                    expandMatrix = np.roll(np.eye(self.z, dtype=int), qij, axis=1)

                qc_matrix[i * self.z: (i + 1) * self.z, j * self.z: (j + 1) * self.z] = expandMatrix
        self.H = qc_matrix
        return qc_matrix


class IEEE80106eQCMatrix(QCMatrix):
    r"""IEEE802.16e的QC-LDPC校验矩阵生成器
    Args:
        - Nbit：码长，可选码率有[576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440,
                                1536, 1632, 1728, 1824, 1920, 2016, 2112, 2208, 2304]
        - rate：码率，可选码率有[1./2, 2./3, 3./4, 5./6]
        - codetype：A/B型号，只有码率为 2/3 或者 3/4 的才可以选择
    """

    def __init__(self, Nbit, rate, codetype: str = None, device='cpu'):
        """
        :self.NonZero: 有三个非0的行数
        """
        self.rateList = [1. / 2, 2. / 3, 3. / 4, 5. / 6]
        self.NbitList = [576, 672, 768, 864, 960, 1056, 1152, 1248, 1344, 1440,
                         1536, 1632, 1728, 1824, 1920, 2016, 2112, 2208, 2304]
        assert rate in self.rateList, '选择了不在标准中的码率'
        assert Nbit in self.NbitList, '选择了不在标准中的码长'
        self.rate = rate

        Hb_index = None
        self.codetype = codetype
        if self.codetype == 'A' or self.codetype is None:
            Hb_index = self.rateList.index(self.rate)
        if self.codetype == 'B':
            Hb_index = self.rateList.index(self.rate) + 3

        # 1/2
        Hb1 = [
            [-1, 94, 73, -1, -1, -1, -1, -1, 55, 83, -1, -1, 7, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, 27, -1, -1, -1, 22, 79, 9, -1, -1, -1, 12, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, 24, 22, 81, -1, 33, -1, -1, -1, 0, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1],
            [61, -1, 47, -1, -1, -1, -1, -1, 65, 25, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, 39, -1, -1, -1, 84, -1, -1, 41, 72, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, 46, 40, -1, 82, -1, -1, -1, 79, 0, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1],
            [-1, -1, 95, 53, -1, -1, -1, -1, -1, 14, 18, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1],
            [-1, 11, 73, -1, -1, -1, 2, -1, -1, 47, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1],
            [12, -1, -1, -1, 83, 24, -1, 43, -1, -1, -1, 51, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1],
            [-1, -1, -1, -1, -1, 94, -1, 59, -1, -1, 70, 72, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1],
            [-1, -1, 7, 65, -1, -1, -1, -1, 39, 49, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
            [43, -1, -1, -1, -1, 66, -1, 41, -1, -1, -1, 26, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
        ]

        # 2/3 A
        Hb2 = [
            [3, 0, -1, -1, 2, 0, -1, 3, 7, -1, 1, 1, -1, -1, -1, -1, 1, 0, -1, -1, -1, -1, -1, -1],
            [-1, -1, 1, -1, 36, -1, -1, 34, 10, -1, -1, 18, 2, -1, 3, 0, -1, 0, 0, -1, -1, -1, -1, -1],
            [-1, -1, 12, 2, -1, 15, -1, 40, -1, 3, -1, 15, -1, 2, 13, -1, -1, -1, 0, 0, -1, -1, -1, -1],
            [-1, -1, 19, 24, -1, 3, 0, -1, 6, -1, 17, -1, -1, -1, 8, 39, -1, -1, -1, 0, 0, -1, -1, -1],
            [20, -1, 6, -1, -1, 10, 29, -1, -1, 28, -1, 14, -1, 38, -1, -1, 0, -1, -1, -1, 0, 0, -1, -1],
            [-1, -1, 10, -1, 28, 20, -1, -1, 8, -1, 36, -1, 9, -1, 21, 45, -1, -1, -1, -1, -1, 0, 0, -1],
            [35, 25, -1, 37, -1, 21, -1, -1, 5, -1, -1, 0, -1, 4, 20, -1, -1, -1, -1, -1, -1, -1, 0, 0],
            [-1, 6, 6, -1, -1, -1, 4, -1, 14, 30, -1, 3, 36, -1, 14, -1, 1, -1, -1, -1, -1, -1, -1, 0],
        ]

        # 3/4 A
        Hb3 = [
            [6, 38, 3, 93, -1, -1, -1, 30, 70, -1, 86, -1, 37, 38, 4, 11, -1, 46, 48, 0, -1, -1, -1, -1],
            [62, 94, 19, 84, -1, 92, 78, -1, 15, -1, -1, 92, -1, 45, 24, 32, 30, -1, -1, 0, 0, -1, -1, -1],
            [71, -1, 55, -1, 12, 66, 45, 79, -1, 78, -1, -1, 10, -1, 22, 55, 70, 82, -1, -1, 0, 0, -1, -1],
            [38, 61, -1, 66, 9, 73, 47, 64, -1, 39, 61, 43, -1, -1, -1, -1, 95, 32, 0, -1, -1, 0, 0, -1],
            [-1, -1, -1, -1, 32, 52, 55, 80, 95, 22, 6, 51, 24, 90, 44, 20, -1, -1, -1, -1, -1, -1, 0, 0],
            [-1, 63, 31, 88, 20, -1, -1, -1, 6, 40, 56, 16, 71, 53, -1, -1, 27, 26, 48, -1, -1, -1, -1, 0],
        ]

        # 5/6
        Hb4 = [
            [1, 25, 55, -1, 47, 4, -1, 91, 84, 8, 86, 52, 82, 33, 5, 0, 36, 20, 4, 77, 80, 0, -1, -1],
            [-1, 6, -1, 36, 40, 47, 12, 79, 47, -1, 41, 21, 12, 71, 14, 72, 0, 44, 49, 0, 0, 0, 0, -1],
            [51, 81, 83, 4, 67, -1, 21, -1, 31, 24, 91, 61, 81, 9, 86, 78, 60, 88, 67, 15, -1, -1, 0, 0],
            [68, -1, 50, 15, -1, 36, 13, 10, 11, 20, 53, 90, 29, 92, 57, 30, 84, 92, 11, 66, 80, -1, -1, 0],
        ]

        # 2/3 B
        Hb5 = [
            [2, -1, 19, -1, 47, -1, 48, -1, 36, -1, 82, -1, 47, -1, 15, -1, 95, 0, -1, -1, -1, -1, -1, -1],
            [-1, 69, -1, 88, -1, 33, -1, 3, -1, 16, -1, 37, -1, 40, -1, 48, -1, 0, 0, -1, -1, -1, -1, -1],
            [10, -1, 86, -1, 62, -1, 28, -1, 85, -1, 16, -1, 34, -1, 73, -1, -1, -1, 0, 0, -1, -1, -1, -1],
            [-1, 28, -1, 32, -1, 81, -1, 27, -1, 88, -1, 5, -1, 56, -1, 37, -1, -1, -1, 0, 0, -1, -1, -1],
            [23, -1, 29, -1, 15, -1, 30, -1, 66, -1, 24, -1, 50, -1, 62, -1, -1, -1, -1, -1, 0, 0, -1, -1],
            [-1, 30, -1, 65, -1, 54, -1, 14, -1, 0, -1, 30, -1, 74, -1, 0, -1, -1, -1, -1, -1, 0, 0, -1],
            [32, -1, 0, -1, 15, -1, 56, -1, 85, -1, 5, -1, 6, -1, 52, -1, 0, -1, -1, -1, -1, -1, 0, 0],
            [-1, 0, -1, 47, -1, 13, -1, 61, -1, 84, -1, 55, -1, 78, -1, 41, 95, -1, -1, -1, -1, -1, -1, 0],
        ]

        # 3/4 B
        Hb6 = [
            [-1, 81, -1, 28, -1, -1, 14, 25, 17, -1, -1, 85, 29, 52, 78, 95, 22, 92, 0, 0, -1, -1, -1, -1],
            [42, -1, 14, 68, 32, -1, -1, -1, -1, 70, 43, 11, 36, 40, 33, 57, 38, 24, -1, 0, 0, -1, -1, -1],
            [-1, -1, 20, -1, -1, 63, 39, -1, 70, 67, -1, 38, 4, 72, 47, 29, 60, 5, 80, -1, 0, 0, -1, -1],
            [64, 2, -1, -1, 63, -1, -1, 3, 51, -1, 81, 15, 94, 9, 85, 36, 14, 19, -1, -1, -1, 0, 0, -1],
            [-1, 53, 60, 80, -1, 26, 75, -1, -1, -1, -1, 86, 77, 1, 3, 72, 60, 25, -1, -1, -1, -1, 0, 0],
            [77, -1, -1, -1, 15, 28, -1, 35, -1, 72, 30, 68, 85, 84, 26, 64, 11, 89, 0, -1, -1, -1, -1, 0],
        ]

        Hb_list = [Hb1, Hb2, Hb3, Hb4, Hb5, Hb6]
        NonZero_list = [5, 4, 3, 1, 6, 2]

        z = Nbit / 24   # 扩展子矩阵长度

        self.NonZero = NonZero_list[Hb_index]      # 有三个非0的行数
        Hb = Hb_list[Hb_index]
        Hb = np.array(Hb, dtype=int)
        mb, nb = Hb.shape
        for i in range(mb):
            for j in range(nb):
                if Hb[i, j] > 0:
                    Hb[i, j] = np.floor(Hb[i, j] * z / 96)
        # Mbit = Nbit - Nbit * self.rate      # 校验位长度
        self.Hb = Hb
        super().__init__(Nbit=Nbit, Kbit=Nbit*self.rate, Hb=Hb, device=device)

    def __repr__(self):
        return f'IEEE802.16e标准的LDPC校验矩阵生成器 [码长：{self.Nbit}, 源消息长：{self.Kbit}, ' \
               f'码率：{self.rate}{", " + self.codetype if self.codetype is not None else ""}]'

    @property
    def make(self):
        return super().make

    def plot_H2(self, matrix_size=None, isSave=None, filename=None, cmap='RdPu',
                dpi=1000, figsize=(7,5), labelsize=8, linewidth=0.5, annot=False):
        """绘制H2的对角线特性"""
        if matrix_size is not None:
            H2 = self.H[:matrix_size[0], self.Kbit:matrix_size[1]+self.Kbit]
        else:
            H2 = self.H[:, self.Kbit:]
        plt.figure(figsize=figsize)
        heatmap = sns.heatmap(H2, cmap=cmap, linewidths=linewidth, annot=annot)
        plt.xticks(rotation=0)  # 旋转x轴刻度标签
        plt.yticks(rotation=0)  # 旋转y轴刻度标签

        heatmap.tick_params(labelsize=labelsize)
        if isSave:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.show()
