import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class HammingCode74:
    def __init__(self):
        # 定义监督位错误模式
        self.error_patterns = np.array([
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

    def encode(self, data_bits):
        """编码 4 位数据位，生成 7 位汉明码"""
        if len(data_bits) != 4:
            raise ValueError("数据位长度必须为 4 位")

        a6, a5, a4, a3 = data_bits
        a0 = a6 ^ a4 ^ a3
        a1 = a6 ^ a5 ^ a3
        a2 = a6 ^ a5 ^ a4

        check_bits = np.array([a2, a1, a0])
        encoded_data = np.concatenate((data_bits, check_bits))
        return encoded_data

    def decode(self, encoded_data):
        """解码 7 位汉明码，检测并纠正错误，返回 4 位数据位"""
        if len(encoded_data) != 7:
            raise ValueError("编码数据长度必须为 7 位")

        a = encoded_data
        S1 = a[0] ^ a[1] ^ a[2] ^ a[4]
        S2 = a[0] ^ a[1] ^ a[3] ^ a[5]
        S3 = a[0] ^ a[2] ^ a[3] ^ a[6]
        SS = np.array([S1, S2, S3])

        for i in range(7):
            if np.array_equal(SS, self.error_patterns[i]):
                a[i] = a[i] ^ 1  # 发现错误，翻转相应的位

        data_bits_corrected = a[:4]
        return data_bits_corrected

def bi_awgn_channel(encoded_data, snr_db):
    """通过双极性 AWGN 通道传输汉明码"""
    snr_linear = 10 ** (snr_db / 10.0)
    noise_std = np.sqrt(1 / (2 * snr_linear))
    noise = noise_std * np.random.randn(*encoded_data.shape)
    received_signal = 2 * encoded_data - 1 + noise
    received_data = np.where(received_signal >= 0, 1, 0)
    return received_data

def simulate_ber(hamming, num_bits, eb_n0_db_range):
    ber_encoded = []
    ber_uncoded = []

    for eb_n0_db in eb_n0_db_range:
        num_errors_encoded = 0
        num_errors_uncoded = 0
        total_bits = 0

        while total_bits < num_bits:
            # 生成随机数据位
            data_bits = np.random.randint(0, 2, 4)

            # 编码
            encoded_data = hamming.encode(data_bits)

            # 通过 BI-AWGN 通道传输（编码）
            received_encoded = bi_awgn_channel(encoded_data, eb_n0_db)

            # 解码（编码）
            decoded_data = hamming.decode(received_encoded)

            # 计算误码数（编码）
            num_errors_encoded += np.sum(data_bits != decoded_data)

            # 通过 BI-AWGN 通道传输（未编码）
            received_uncoded = bi_awgn_channel(data_bits, eb_n0_db)

            # 计算误码数（未编码）
            num_errors_uncoded += np.sum(data_bits != received_uncoded)

            total_bits += 4

        ber_enc = num_errors_encoded / total_bits
        ber_unc = num_errors_uncoded / total_bits

        ber_encoded.append(ber_enc if ber_enc > 0 else np.nan)
        ber_uncoded.append(ber_unc if ber_unc > 0 else np.nan)

    return ber_encoded, ber_uncoded

# 仿真参数
num_bits = 1e5  # 每个信噪比下的比特数
eb_n0_db_range = np.arange(0, 11, 1)  # 信噪比 (Eb/N0) 从 0 到 10 dB

# 初始化汉明码类
hamming = HammingCode74()

# 仿真 BER
ber_encoded, ber_uncoded = simulate_ber(hamming, int(num_bits), eb_n0_db_range)

# 绘制 BER 性能曲线
plt.figure()
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('bmh')
plt.semilogy(eb_n0_db_range, ber_encoded, 'o-', label='编码系统 (7,4) 汉明码')
plt.semilogy(eb_n0_db_range, ber_uncoded, 's-', label='未编码系统')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('BER')
plt.title('BI-AWGN 信道下的 BER 性能比较')
plt.legend()
plt.grid(True)
# plt.ylim([1e-5, 1])
plt.savefig('hamming.jpg', dpi=500)
plt.show()
