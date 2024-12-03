import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.tool as tool
from LDPC.decode.LDPC_decode import *
from LDPC.encode.LDPC_encode import *
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from copy import *

# 实验参数设置
# data_num = 3
data_num4 = 10
data_num3 = 20
data_num2 = 20
data_num1 = 40
# data_num4 = 20
# data_num3 = 40
# data_num2 = 40
# data_num1 = 80
method = 'LLR-BP'
# SNR_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# SNR_list1 = [-5, -4, -3.01, -2, -1, 0, 1, 2, 3, 4, 5, 6,7,8]
# SNR_list1 = [-5, -4, -3.01, -2, -1, 0, 1, 2, 3, 4, 5, 6,7,8]
# SNR_list2 = [-1.761, -0.761, 0.239, 1.239, 2.239, 3.239, 4.239, 5.239, 6.239, 7.239,8.239,9.239]
# SNR_list3 = [-1.25, -0.25, 0.75, 1.75, 2.75, 3.75, 4.75, 5.75, 6.75, 7.75,8.75,9.75]
# SNR_list4 = [-1.792, -0.792, 0.208, 1.208, 2.208, 3.208, 4.208, 5.208, 6.208, 7.208,8.208, 9.208, 10.208]
# SNR_list1 = [-0.5, 0, 0.5, 1,1.5, 2, 3, 4, 5, 6,7,8]
# SNR_list2 = [-0.5, 0, 0.5, 1,1.5, 2, 3, 4, 5, 6,7,8]
# SNR_list3 = [-0.5, 0, 0.5, 1,1.5, 2, 3, 4, 5, 6,7,8]
# SNR_list4 = [-0.5, 0, 0.5, 1,1.5, 2, 3, 4, 5, 6,7,8]
SNR_list1 = [-4, -3.5, -3, -2.5, -2, -1.5, -1, 0, 1, 2, 3, 4, 5, 6,7,8]
SNR_list2 = [-4, -3.5, -3, -2.5, -2, -1.5, -1, 0, 1, 2, 3, 4, 5, 6,7,8]
# SNR_list3 = [-5, -4, -4.5, -3, -3.5, -2, -1, 0, 1, 2, 3, 4, 5, 6,7,8]
SNR_list4 = [-4, -3.5, -3, -2.5, -2, -1.5, -1, 0, 1, 2, 3, 4, 5, 6,7,8]
#
#
ebn0_data = []
#
# 实例化一个矩阵构造器
Nbit1 = 576
Rate1 = 1 / 2
H_constructor = IEEE80106eQCMatrix(Nbit=Nbit1, rate=Rate1, codetype='A')
# 实例化一个仿真链路
my_link = tool.Link(H_constructor, encoder='QC-LDPC Encoder', decoder='LDPC Decoder',
                    channel='Bi-AWGN', modem='BPSK')
# 设置想要仿真的编解码方法
encode_method = ['Quasi Cyclic Bidiagonal Fast encode']
decode_method = [method]

# 仿真
# BER1, SNR1 =  my_link.simulate_BER(encode_method, decode_method, data_num=data_num1, is_save=False, imagID=1,
#                                    save_dir='/Users/mac/Desktop/LDPC-star/imag',
#                                    SNR=SNR_list1)
BER1, SNR1 = my_link.simulate_BER(encode_method, decode_method, data_num=data_num1, is_save=False, imagID=1,
                                   save_dir='/Users/mac/Desktop/LDPC-star/imag',
                                   SNR=SNR_list1)
print(f'BER:{BER1}')
ebn0_data.append(deepcopy(SNR1))
# BER1 = [[0.152, 0.141, 0.094, 0.04, 0.0052, 0.0011, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
print(SNR1)
print(f'BER:{BER1}')
#
#
#
# 实例化一个矩阵构造器
Nbit2 = 1440
Rate2 = 1 / 2
H_constructor = IEEE80106eQCMatrix(Nbit=Nbit2, rate=Rate2, codetype='A')
# 实例化一个仿真链路
my_link = tool.Link(H_constructor, encoder='QC-LDPC Encoder', decoder='LDPC Decoder',
                    channel='Bi-AWGN', modem='BPSK')
# 设置想要仿真的编解码方法
encode_method = ['Quasi Cyclic Bidiagonal Fast encode']
decode_method = [method]

# 仿真
BER2, SNR2 =  my_link.simulate_BER(encode_method, decode_method, data_num=data_num2, is_save=False, imagID=1,
                                   save_dir='/Users/mac/Desktop/LDPC-star/imag',
                                   SNR=SNR_list2)
print(f'BER:{BER2}')
ebn0_data.append(deepcopy(SNR2))
# BER2 = [[0.153, 0.133, 0.093, 0.033, 0.0026, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
print(SNR2)
print(f'BER:{BER2}')
#
#
# # 实例化一个矩阵构造器
# Nbit3 = 1728
# Rate3 = 1 / 2
# H_constructor = IEEE80106eQCMatrix(Nbit=Nbit3, rate=Rate3, codetype='A')
# # 实例化一个仿真链路
# my_link = tool.Link(H_constructor, encoder='QC-LDPC Encoder', decoder='LDPC Decoder',
#                     channel='Bi-AWGN', modem='BPSK')
# # 设置想要仿真的编解码方法
# encode_method = ['Quasi Cyclic Bidiagonal Fast encode']
# decode_method = [method]
#
# # 仿真
# BER3, SNR3 = my_link.simulate_BER(encode_method, decode_method, data_num=0, is_save=False, imagID=1,
#                                    save_dir='/Users/mac/Desktop/LDPC-star/imag',
#                                    SNR=SNR_list3)
# print(SNR3)
# ebn0_data.append(deepcopy(SNR3))
# BER3 = [[0.14718364197530864, 0.12191358024691358, 0.09143518518518519, 0.02333719135802469, 0.0, 0, 0, 0, 0, 0, 0, 0]]
# print(f'BER:{BER3}')
#
# 实例化一个矩阵构造器
Nbit4 = 2304
Rate4 = 1 / 2
H_constructor = IEEE80106eQCMatrix(Nbit=Nbit4, rate=Rate4, codetype='A')
# 实例化一个仿真链路
my_link = tool.Link(H_constructor, encoder='QC-LDPC Encoder', decoder='LDPC Decoder',
                    channel='Bi-AWGN', modem='BPSK')
# 设置想要仿真的编解码方法
encode_method = ['Quasi Cyclic Bidiagonal Fast encode']
decode_method = [method]

# 仿真
BER4, SNR4 =  my_link.simulate_BER(encode_method, decode_method, data_num=data_num4, is_save=False, imagID=1,
                                   save_dir='/Users/mac/Desktop/LDPC-star/imag',
                                   SNR=SNR_list4)
print(SNR4)
ebn0_data.append(deepcopy(SNR4))
# BER4 = [[0.1423611111111111, 0.1175733024691358, 0.09042515432098765, 0.019579475308641976, 0.0, 0, 0, 0, 0, 0, 0, 0]]
print(f'BER:{BER4}')
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# # def plot_ber_vs_ebn0(ber_data, ebn0_data, rate_labels, save_path=None):
# #     """
# #     绘制不同码率的 LDPC 码的 BER 与 Eb/N0 的关系图。
# #
# #     参数:
# #     ber_data : list of np.ndarray
# #         包含四个 BER np.ndarray 的列表，每个 ndarray 形状为 (1, n)。
# #     ebn0_data : list of list of float
# #         每个编码方案对应的 Eb/N0 的列表。
# #     rate_labels : list of str
# #         码率的标签列表，例如 ['1/2', '2/3', '3/4', '5/6']。
# #     save_path : str, optional
# #         如果提供，图像将保存到此路径。
# #     """
# #     plt.style.use('bmh')
# #     plt.figure(figsize=(10, 6))
# #
# #     # 遍历 BER 数据并绘制每个码率的曲线
# #     for ber, ebn0, label in zip(ber_data, ebn0_data, rate_labels):
# #         plt.plot(ebn0, ber.flatten(), label=f'Rate {label}', marker='o')
# #
# #     plt.yscale('log')  # 将 y 轴设为对数刻度
# #     plt.xlabel('Eb/N0 (dB)')
# #     plt.ylabel('BER')
# #     plt.title('BER vs Eb/N0 for different LDPC rates')
# #     plt.legend()
# #     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# #
# #     if save_path is not None:
# #         plt.savefig(save_path, dpi=500)
# #     plt.show()

def plot_ber_vs_ebn0(ber_data, ebn0_data, rate_labels, save_path=None):
    """
    绘制不同码率的 LDPC 码的 BER 与 Eb/N0 的关系图。

    参数:
    ber_data : list of np.ndarray
        包含四个 BER np.ndarray 的列表，每个 ndarray 形状为 (1, n)。
    ebn0_data : list of list of float
        每个编码方案对应的 Eb/N0 的列表。
    rate_labels : list of str
        码率的标签列表，例如 ['1/2', '2/3', '3/4', '5/6']。
    save_path : str, optional
        如果提供，图像将保存到此路径。
    """
    plt.style.use('bmh')
    plt.figure(figsize=(10, 6))

    # 遍历 BER 数据并绘制每个码率的曲线
    for ber, ebn0, label in zip(ber_data, ebn0_data, rate_labels):
        # 去掉 BER 为 0 的数据
        ber = ber.flatten()
        ebn0 = np.array(ebn0)
        mask = ber > 0
        ber = ber[mask]
        ebn0 = ebn0[mask]

        plt.plot(ebn0, ber, label=f'codeLength {label}', marker='o')

    plt.yscale('log')  # 将 y 轴设为对数刻度
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.title('BER vs Eb/N0 for different LDPC codeLength')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    if save_path is not None:
        plt.savefig(save_path, dpi=500)
    plt.show()
#
# BER1 = [[0.152, 0.141, 0.094, 0.04, 0.0052, 0.0011, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
# BER2 = [[0.153, 0.133, 0.093, 0.033, 0.0026, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
# BER3 = [[0.14718364197530864, 0.12591358024691358, 0.09143518518518519, 0.02333719135802469, 0.0, 0, 0, 0, 0, 0, 0, 0]]
# BER4 = [[0.1423611111111111, 0.1205733024691358, 0.08942515432098765, 0.019579475308641976, 0.0, 0, 0, 0, 0, 0, 0, 0]]


# 示例用法
ber_data = [
    # np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6]).reshape(1, -1),
    # np.array([1.5e-2, 1.2e-3, 1.1e-4, 0.9e-5, 0.7e-6]).reshape(1, -1),
    # np.array([2e-2, 1.8e-3, 1.5e-4, 1.1e-5, 1.0e-6]).reshape(1, -1),
    # np.array([3e-2, 2.5e-3, 2.0e-4, 1.5e-5, 1.3e-6]).reshape(1, -1)
    np.array(BER1),
    np.array(BER2),
    # np.array(BER3),
    np.array(BER4)
]

# ebn0_data = [
#     [2.5, 3, 3.5, 4,4.5, 5, 6, 7, 8, 9,10,11],
#     [2.5, 3, 3.5, 4,4.5, 5, 6, 7, 8, 9,10,11],
#     [2.5, 3, 3.5, 4,4.5, 5, 6, 7, 8, 9,10,11],
#     [2.5, 3, 3.5, 4,4.5, 5, 6, 7, 8, 9,10,11],
# ]

print(SNR1)
print(SNR2)
# print(SNR3)
print(SNR4)

print(BER1)
print(BER2)
print(BER4)
# rate_labels = ['576', '1152', '1728', '2304']
rate_labels = ['576', '1440','2304']

plot_ber_vs_ebn0(ber_data, ebn0_data, rate_labels, save_path='码长实验3（LLR）.jpg')


