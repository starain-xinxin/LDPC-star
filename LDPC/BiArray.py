import numpy as np
from typing import Union

def inv_bin(in_matrix):
    """
    一个二元域矩阵求逆的函数

    - TODO：但对于输入的矩阵是否是可逆的问题，无法回答，无法报错，谨慎使用
    """
    # 计算输入矩阵的行数和列数
    m, n = in_matrix.shape
    if m != n:
        print('m != n')
        return None

    # 初始化单位矩阵
    E = np.eye(m, dtype=int)

    # 做行变换，变成下三角阵
    for i in range(m):
        none_zeros_index = np.where(in_matrix[:, i])[0]  # 第 i 列非零元素的行索引
        none_zeros_index = none_zeros_index[none_zeros_index >= i]  # 过滤掉前面的零元素
        if len(none_zeros_index) == 0:  # 如果该列全为零
            rand_index = np.random.randint(i+1, m+1)  # 随机生成一个索引
            # 列交换
            in_matrix[:, [i, rand_index]] = in_matrix[:, [rand_index, i]]
            # E 交换
            E[:, [i, rand_index]] = E[:, [rand_index, i]]
        id1 = none_zeros_index[0]  # 第一个非零元素的索引
        # in_matrix 交换
        in_matrix[[i, id1]] = in_matrix[[id1, i]]
        # E 交换
        E[[i, id1]] = E[[id1, i]]

        none_zeros_index = np.where(in_matrix[:, i])[0]  # 第 i 列非零元素的行索引
        for cc in none_zeros_index:
            if cc != i:
                in_matrix[cc] = np.mod(in_matrix[cc] + in_matrix[i], 2)
                E[cc] = np.mod(E[cc] + E[i], 2)

    return E


class BiArray:
    """
    二元域的张量的实现，重载了如下运算符

    - +: 二进制的加法
    - -：矩阵的每个元素取反
    - ～：矩阵求逆
    - @：矩阵乘法

    同样，也是支持张量切片操作的。
    """

    def __init__(self, array: Union[list, np.ndarray, 'BiArray']):
        if isinstance(array, BiArray):
            self.array = array.array
            self.shape = array.shape
        else:
            self.array = np.array(array, dtype=int)
            self.shape = self.array.shape

    def __repr__(self):
        return f'BiArray, shape:{self.shape} \n ' + f'{self.array} \n'

    def __add__(self, other: 'BiArray'):
        result = self.array + other.array
        result = np.mod(result, 2)
        return BiArray(result)

    def __neg__(self):
        """ - 号重载为每个元素取反 """
        result = self.array+1
        result = np.mod(result, 2)
        return BiArray(result)

    def __invert__(self):
        """ ~ 号重载为矩阵求逆 """
        # TODO:
        result = self.inv()
        return BiArray(result)

    def __matmul__(self, other: 'BiArray'):
        result = self.array @ other.array
        result = np.mod(result, 2)
        return BiArray(result)

    def __getitem__(self, indices):
        # 处理切片操作
        result = self.array[indices]
        return BiArray(result)

    def __setitem__(self, index, value:'BiArray'):
        # 处理切片操作
        self.array[index] = value.array

    def __eq__(self, other:'BiArray'):
        return self.array == other.array

    def transpose(self, indices=None):
        result = self.array.transpose(indices)
        return BiArray(result)

    @property
    def T(self):
        """
        二元域张量转置

        - 此方法可以当作属性调用
        - 无论是多少维张量，均是对于最后两个维度进行交换(这与numpy不同)
        """
        dim_num = len(self.shape)
        index = [i for i in range(dim_num)]
        index[-1] = index[-1] - 1
        index[-2] = index[-2] + 1
        result = self.array.transpose(index)
        return BiArray(result)

    def inv(self):
        # 计算输入矩阵的行数和列数
        return inv_bin(self.array)

    @property
    def numpy(self):
        """ 变为ndarray """
        return self.array

    def reshape(self, shape:Union[int, tuple]):
        result = self.array.reshape(shape)
        return BiArray(result)
