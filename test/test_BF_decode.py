#############################################################################################################################
#                                           测试BF_decode函数
#
#   测试资料：https://zhuanlan.zhihu.com/p/514670102
#   结果：通过
#
#############################################################################################################################
from LDPC.decode.LDPC_decode import *

Hb = [
    [1, 1, 0, 1, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [1, 0, 1, 0, 0, 1]
]
Hb = np.array(Hb)
Hb = Hb-1

H_constructor = QCMatrix(6,3,Hb)

decoder = LdpcDecoder(H_constructor)

code = [0, 1, 0, 0, 1, 1]

code1, flag = decoder.BF_decode(code)
print(code1)

print(flag)
