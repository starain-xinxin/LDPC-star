#############################################################################################
#
#                               全链路仿真接口使用示例
#
#############################################################################################
import utils.tool as tool
from LDPC.decode.LDPC_decode import *
from LDPC.encode.LDPC_encode import *
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 设置实验参数
imagID = 4

# 设置IEEE 802.16e协议下的参数，码字与码率
Nbit = 768
Rate = 1 / 2

# 实例化一个矩阵构造器
H_constructor = IEEE80106eQCMatrix(Nbit=Nbit, rate=Rate, codetype='A')

# 实例化一个全仿真链路
my_link = tool.Link(H_constructor, encoder='QC-LDPC Encoder', decoder='LDPC Decoder',
                    channel='Bi-AWGN', modem='BPSK')

# 设置想要仿真的编解码方法
encode_method = ['Quasi Cyclic Bidiagonal Fast encode']
decode_method = ['None', 'BF',  'WBF', 'SPA', 'LLR-BP']
# decode_method = ['BF', 'None', 'WBF']

# 仿真
my_link.simulate_BER(encode_method, decode_method, data_num=100, save_dir='/Users/mac/Desktop/LDPC-star/imag',
                     is_save=True, imagID=imagID,
                     SNR=[-3.01, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10])
