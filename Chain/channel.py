import Chain.modem
import numpy as np
import LDPC

class Channel:
    def __init__(self, modem: Chain.modem.Modem):
        self.modem = modem
        pass

    def __repr__(self):
        return f'Base class: channel'

    def forward(self, code):
        """
        信道传输模拟函数
        :param code: BiArray [n,1]
        :return: code: BiArray [n,1]
        """
        pass

    def reset(self, SNR):
        """ 重置信道参数 """
        raise NotImplementedError

    def initialize_llr(self, received_signal: np.ndarray):
        raise NotImplementedError


class BiAwgn(Channel):
    """
    Bi-AWGN:二元加性白高斯信道
    """

    def __init__(self, modem: Chain.modem.Modem, std=1, mean=0):
        super().__init__(modem)
        self.std = std
        self.mean = mean

    def __repr__(self):
        return f'信道：Bi-AWGN\n调制方式：{self.modem.__repr__()} sigma:{self.std}, mean:{self.mean}'

    def forward(self, code: LDPC.BiArray, is_soft=False):
        """
        :param code     输入的编码   BiArray [n,1]
        :param is_soft  是否需要软输出 ndarray [n,1]  默认硬输出
        """
        if not is_soft:                                 # 硬输出
            symbol = self.modem.Bit2Symbol(code)
            symbol_with_noise = self.AddNoise(symbol)
            code = self.modem.decide(symbol_with_noise)
            code = self.modem.Symbol2Bit(code)
        else:                                           # 软输出
            symbol = self.modem.Bit2Symbol(code)
            code = self.AddNoise(symbol)
        return code

    def reset(self, SNR):
        # 信号功率
        signal_power = self.modem.maxV**2
        # 将SNR从dB转换为线性值
        snr_linear = 10 ** (SNR / 10)
        # 计算噪声功率
        noise_power = signal_power / snr_linear
        # 计算标准差
        noise_std = np.sqrt(noise_power / 2)
        self.std = noise_std

    def AddNoise(self, code: np.ndarray):
        noise = np.random.normal(self.mean, self.std, code.shape)
        code = code + noise
        return code

    def initialize_llr(self, received_signal: np.ndarray):
        """ 从接收信号初始化 LLR 值"""
        if isinstance(self.modem, Chain.modem.BPSK):
            return -2 * received_signal / (self.std ** 2)
        else:
            raise NotImplementedError
