# 一个LDPC编解码库，并实现GPU的加速
## 来源
这个仓库的实际是复旦大学2024年春季学期《通信编码原理与技术》的课程设计
## 代码结构
``` txt
-- LDPC-star
    |- MatrixConstructor 矩阵构造器
        |- HMatrixConstructor.py  校验矩阵构造器
    |- LDPC LDPC的核心编解码实现
        ｜- encode LDPC编码器实现
        ｜- decode LDPC解码器实现
        ｜- BiArray 二元有限域的张量实现
    ｜- Chain 信道仿真实现
        ｜- channel.py 各种信道实现
        ｜- modem.py 各种调制解调器实现
    ｜- utils
        ｜- tool.py 全链路仿真的工具
    ｜- csource GPU加速实现
    ｜- example 一些使用这些库的范例文件
    ｜- imag 放置一些结果图
```

## 正在施工。。。
