from LDPC.decode.LDPC_decode import *
from LDPC.encode.LDPC_encode import *
from MatrixConstructor.HMatrixConstructor import *
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from Chain.channel import *
import math


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

    def simulate_BER(self, encode_method: Union[list, str], decode_method: Union[list, str],
                     data_num=500, SNR: list = None,
                     save_dir=None, figsize=(10.5,7), dpi=500):
        """
        仿真函数:对于固定编码标准，对于多种编解码方式仿真在不同的SNR中的BER，并且
        :param figsize:图片大小
        :param save_dir:图片存储文件夹
        :param encode_method:编码方案字符串列表
        :param decode_method:解码方案字符串列表
        :param data_num:随机采样的码字数量
        :param SNR:仿真信噪比列表
        :param dpi:保存图像的dpi
        :return:误码率list
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
        BER = []
        for en_method in encode_method:
            for de_method in decode_method:
                print(f'编码方法：{en_method}, 解码方法：{de_method}')
                error_bit_list = []
                for snr in SNR:
                    self.channel.reset(snr)
                    error_bit = 0

                    for x in tqdm(data, desc=f'SNR={snr}'):
                        # 编码 -->  信道 -->  解码
                        code = self.encoder.encode(x, en_method, isVal=self.isVal)
                        code = self.channel.forward(code)
                        code = self.decoder.decode(code, de_method)

                        # 计算误码个数
                        error_bit = error_bit + count_mismatch_elements(x, code)

                    # 计算在当前SNR下的平均误bit数
                    error_bit_list.append(error_bit / (K * data_num))

                # 统计所有SNR下的误bit率
                BER.append(error_bit_list)

        # 画图
        self.plot_BER(BER, SNR, encode_method, decode_method, save_dir, figsize=figsize, dpi=dpi)
        return BER

    @staticmethod
    def plot_BER(BER, SNR, encode_method, decode_method, save_dir, figsize=(10, 8), dpi=500):
        sns.set_theme(style="darkgrid", palette="pastel")
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        min_snr = min(SNR)
        max_snr = max(SNR)
        interval = math.floor((max_snr - min_snr) / len(SNR) * 2) / 2
        i = 0
        j = 0
        for en_method in encode_method:
            plt.figure(figsize=figsize)
            for de_method in decode_method:
                data = {'SNR': SNR, 'BER': BER[i * len(encode_method) + j]}
                df = pd.DataFrame(data)
                sns.lineplot(x='SNR', y='BER', data=df, label=f'{de_method}', linewidth=2.5)
                j = j + 1
            plt.yscale('log')
            plt.title(f'{en_method}编码方案下，各种解码方案的误比特率')
            plt.grid(which='major', linestyle='--', linewidth='0.3', color='gray')
            plt.grid(which='minor', linestyle='-.', linewidth='0.25', color='gray')
            plt.xticks(np.arange(min_snr, max_snr, interval))  # 从0到60，间隔为5
            plt.savefig(save_dir + f'/BER-SNR:{en_method}编码方案.jpeg', dpi=dpi, bbox_inches='tight')
            plt.show()
            i = i + 1


def count_mismatch_elements(vector1, vector2):
    # 检查向量是否具有相同的形状
    if vector1.shape != vector2.shape:
        raise ValueError("Vectors must have the same shape")
    # 对比两个向量，获取不一致元素的布尔掩码
    mismatch_mask = vector1 != vector2
    # 统计不一致元素的数量
    mismatch_count = np.sum(mismatch_mask)
    return mismatch_count
