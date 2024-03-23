from LDPC.decode.LDPC_decode import *
from LDPC.encode.LDPC_encode import *
from MatrixConstructor.HMatrixConstructor import *
from typing import Union
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from Chain.channel import *


class Link:
    def __init__(self, matrix_constructor: Optional[HMatrixConstructor],
                 encoder: Union[str, Encoder], decoder: Union[str, Decoder],
                 channel: Union[Channel, str], modem: [str, Modem], isVal=True):
        # 初始化矩阵构造器
        self.matrix_constructor = matrix_constructor

        # 初始化编码器
        if isinstance(encoder, Encoder):
            self.encoder = encoder
        else:
            if encoder == 'QC-LDPC Encoder':
                self.encoder = QCLdpcEncoder(self.matrix_constructor)
            else:
                assert 1, '编码器初始化错误'

        # 初始化解码器
        if isinstance(decoder, Decoder):
            self.decoder = decoder
        else:
            if decoder == 'LDPC Decoder':
                self.decoder = LdpcDecoder(self.matrix_constructor)
            else:
                assert 1, '解码器初始化错误'

        # 初始化调制解调器
        if isinstance(modem, Modem):
            self.modem = modem
        else:
            if modem == 'BPSK':
                self.modem = BPSK()
            else:
                assert 1, '调制解调器初始化错误'

        # 初始化信道
        if isinstance(channel, Channel):
            self.channel = channel
        else:
            if channel == 'Bi-AWGN':
                self.channel = BiAwgn(self.modem)
            else:
                assert 1, '信道初始化错误'

        # 编码验证
        self.isVal = isVal

    def __repr__(self):
        return f'矩阵构造器：{self.matrix_constructor.__repr__()}\n' \
               f'编码器：{self.encoder.__repr__()}\n' \
               f'解码器：{self.decoder.__repr__()}\n' \
               f'信道：{self.channel.__repr__()}\n'

    def simulate(self, encode_method: Union[list, str], decode_method: Union[list, str],
                 save_dir=None, data_num=500, SNR: list = None):
        """
        仿真函数
        :param save_dir:
        :param encode_method:
        :param decode_method:
        :param data_num:随机采样的码字数量
        :param SNR:仿真信噪比列表
        :return:
        """
        # 编解码方法列表化
        if isinstance(encode_method, str):
            encode_method = [encode_method]
        if isinstance(decode_method, str):
            decode_method = [decode_method]

        # SNR
        if SNR is None:
            SNR = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.4, 1.8, 2.2, 2.6, 3, 3.5, 4, 5, 6, 7.5, 9, 12]

        # 生成data
        K = self.matrix_constructor.Kbit
        data = np.random.randint(2, size=(data_num, K))

        # 仿真
        tt_encode = []
        tt_decode = []
        BER = []
        for en_method in encode_method:
            for de_method in decode_method:
                print(f'编码方法：{en_method}, 解码方法：{de_method}')
                error_bit_list = []
                t1 = []
                t2 = []
                for snr in SNR:
                    self.channel.reset(snr)
                    error_bit = 0

                    for x in tqdm(data, desc=f'SNR={snr}'):
                        # 编码 -->  信道 -->  解码
                        t1_start_time = time.time()
                        code = self.encoder.encode(x, en_method, isVal=self.isVal)
                        t1.append(time.time() - t1_start_time)

                        code = self.channel.forward(code)

                        t2_start_time = time.time()
                        code = self.decoder.decode(code, de_method)
                        t2.append(time.time() - t2_start_time)

                        # 计算误码个数
                        error_bit = error_bit + count_mismatch_elements(x, code)
                    # 计算在当前SNR下的平均误bit数
                    error_bit_list.append(error_bit / (K * data_num))
                # 统计所有SNR下的误bit率
                BER.append(error_bit_list)
                # 对于某个编解码方法绘制SNR-BER图

                # 对于某个编解码方法求出编解码速度(ms/code)
                tt_encode.append(sum(t1) / len(t1) * 1000)
                tt_decode.append(sum(t2) / len(t2) * 1000)
                # 绘制编解码方法对应的速度
        return SNR, encode_method, decode_method, tt_encode, tt_decode, BER


def count_mismatch_elements(vector1, vector2):
    # 检查向量是否具有相同的形状
    if vector1.shape != vector2.shape:
        raise ValueError("Vectors must have the same shape")
    # 对比两个向量，获取不一致元素的布尔掩码
    mismatch_mask = vector1 != vector2
    # 统计不一致元素的数量
    mismatch_count = np.sum(mismatch_mask)
    return mismatch_count
