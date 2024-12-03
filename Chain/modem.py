from LDPC.BiArray import *


class Modem:
    def __init__(self, maxV=1):
        self.maxV = maxV
        pass

    def __repr__(self):
        pass

    def Bit2Symbol(self, code):
        """
        调制函数
        :param code: BiArray [n,1]
        :return: code: ndarray(np.float) [symbols,1]
        """
        pass

    def decide(self,code):
        """
        判决函数
        :param code: 接受的dtype=np.float ndarray [symbols,1]
        :return 判决后的symbol ndarray(np.float)  [symbols,1]
        """
        pass

    def Symbol2Bit(self, code):
        """
        解调
        :param code: 接受的dtype=np.float ndarray [symbols,1]
        :return: receive_code: BiArray  [n,1]
        """
        pass


class BPSK(Modem):
    """ BPSK的调制解调器 """

    def __init__(self, maxV=1):
        super().__init__(maxV)

    def __repr__(self):
        return f'BPSK, 最大电压{self.maxV}'

    def Bit2Symbol(self, code: BiArray):
        code = code.numpy  # ndarray [n, 1]
        code = np.where(code == 0, -self.maxV, code)
        code = np.where(code == 1, self.maxV, code)
        return code.astype(float)

    def decide(self, code: np.ndarray):
        code = np.where(code > 0, 1, code)
        code = np.where(code <= 0, 0, code)
        return code.astype(int)

    def Symbol2Bit(self, code:np.ndarray):
        return BiArray(code)
