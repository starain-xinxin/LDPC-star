from LDPC.decode.LDPC_decode import *
from LDPC.encode.LDPC_encode import *
from MatrixConstructor.HMatrixConstructor import *
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from Chain.channel import *
from Chain.modem import *
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

        # 码率
        self.rate = self.encoder.Nbit / self.encoder.Kbit

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
                     data_num=500, SNR: list = None, is_Eb_N0=True,
                     save_dir=None, figsize=(10.5, 7), dpi=500, is_save=False, imagID=0):
        """
        仿真函数:对于固定编码标准，对于多种编解码方式仿真在不同的SNR中的BER，并且
        :param is_Eb_N0:
        :param imagID:
        :param is_save:是否保存图片
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
            SNR = [0, 1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 10]

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
                        # TODO: 这里要想一个办法适应软硬判决
                        if de_method == 'LLR-BP' or de_method == 'WBF' or de_method == 'SPA' \
                                or de_method == 'MSA':
                            code = self.channel.forward(code, is_soft=True)
                        else:
                            code = self.channel.forward(code, is_soft=False)
                        code = self.decoder.decode(code, de_method, channel=self.channel)

                        # 计算误码个数
                        error_bit = error_bit + count_mismatch_elements(x, code)

                    # 计算在当前SNR下的平均误bit数
                    error_bit_list.append(error_bit / (K * data_num))
                    if error_bit / (K * data_num) == 0:
                        break
                # 填充未仿真的SNR位置为0
                error_bit_list.extend([0] * (len(SNR) - len(error_bit_list)))
                # 统计所有SNR下的误bit率
                BER.append(error_bit_list)

        # 画图
        if is_Eb_N0:
            print(self.rate)
            for i in range(len(SNR)):
                SNR[i] += 10*math.log10(self.rate)
            self.plot_BER(BER, SNR, encode_method, decode_method, save_dir, figsize=figsize, dpi=dpi, is_save=is_save,
                          imagID=imagID)
        else:
            raise NotImplementedError
        return BER, SNR

    @staticmethod
    def plot_BER(BER, SNR, encode_method, decode_method, save_dir, figsize=(11, 7), dpi=500, is_save=False, imagID=0):
        sns.set_theme(style="whitegrid", palette="muted")
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
                # data = {'SNR': SNR, 'BER': BER[i * len(encode_method) + j]}
                # df = pd.DataFrame(data)
                # 修改开始：过滤 BER 等于 0 的数据点
                data = {'Eb/N0': SNR, 'BER': BER[i * len(encode_method) + j]}
                df = pd.DataFrame(data)
                df = df[df['BER'] > 0]
                # 修改结束
                sns.lineplot(x='Eb/N0', y='BER', data=df, label=f'{de_method}', linewidth=1.8)
                plt.scatter(df['Eb/N0'], df['BER'], s=17, zorder=3)
                j = j + 1
            plt.yscale('log')
            plt.title(f'{en_method}编码方案下，各种解码方案的误比特率')
            plt.grid(which='major', linestyle='--', linewidth='0.3', color='gray')
            plt.grid(which='minor', linestyle='-.', linewidth='0.25', color='gray')
            plt.xticks(np.arange(min_snr, max_snr, interval))  # 从0到60，间隔为5
            if is_save:
                plt.savefig(save_dir + f'/Eb-N0-SNR:{en_method}编码方案-{imagID}.jpeg', dpi=dpi, bbox_inches='tight')
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
