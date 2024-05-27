#############################################################################################################################
#                                           测试SPA_decode函数
#
#   结果：
#
#############################################################################################################################
import Chain
import MatrixConstructor as mt
import LDPC
import numpy as np
import utils

# 1.设置IEEE 802.16e协议下的参数，码字与码率, SNR
Nbit = 768
Rate = 1 / 2
SNR = 0

# 2.实例化一个矩阵构造器
H_constructor = mt.IEEE80106eQCMatrix(Nbit=Nbit, rate=Rate, codetype='A')

# 3.实例化 编码器 解码器
ldpc_encoder = LDPC.QCLdpcEncoder(H_constructor)
ldpc_decoder = LDPC.LdpcDecoder(H_constructor)

# 4.初始化 调制解调器 与 信道
modem = Chain.modem.BPSK()
channel = Chain.channel.BiAwgn(modem)

# 5.仿真验证
# 伪造数据
K = H_constructor.Kbit
data = np.random.randint(2, size=(K,))
# 编码
code = ldpc_encoder.encode(data, isVal=True)
# 传播，接收
channel.reset(SNR)

# 解码
code_ = channel.forward(code, is_soft=False)
code1 = ldpc_decoder.decode(code_, 'BF', channel=channel, display=True, max_iter=20)
print(utils.count_mismatch_elements(data, code1))
code__ = channel.forward(code, is_soft=True)
code2 = ldpc_decoder.decode(code__, 'SPA', channel=channel, display=True, max_iter=20)
print(utils.count_mismatch_elements(data, code2))
# 统计误码数

