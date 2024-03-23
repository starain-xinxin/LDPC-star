import utils.tool as tool
from LDPC.decode.LDPC_decode import *
from LDPC.encode.LDPC_encode import *
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

Nbit = 576
Rate = 3 / 4

H_constructor = IEEE80106eQCMatrix(Nbit=Nbit, rate=Rate, codetype='B')

my_link = tool.Link(H_constructor, encoder='QC-LDPC Encoder', decoder='LDPC Decoder', channel='Bi-AWGN', modem='BPSK')

encode_method = ['Quasi Cyclic Bidiagonal Fast encode']
decode_method = ['BF']

SNR, encode_method, decode_method, tt1, tt2, BER = my_link.simulate(encode_method, decode_method, data_num=500)

decode_method1 = ['None']

SNR, encode_method, decode_method, tt1, tt2, BER1 = my_link.simulate(encode_method, decode_method1, data_num=500)

sns.set_theme(style="darkgrid", palette="pastel")
# 创建一个示例数据框
data = {'SNR':SNR, 'BER': BER[0]}
df = pd.DataFrame(data)
data2 = {'SNR':SNR, 'BER': BER1[0]}
df2 = pd.DataFrame(data2)

# 绘制折线图
sns.lineplot(x='SNR', y='BER', data=df, label='BF')
sns.lineplot(x='SNR', y='BER', data=df2, label='None')
plt.yscale('log')
plt.title(f'simulate')
plt.show()

#
# plt.plot(SNR, BER[0], label='BF')
# plt.plot(SNR, BER1[0], label='None')
# plt.legend()
# plt.yscale('log')
# plt.title('')
# plt.show()