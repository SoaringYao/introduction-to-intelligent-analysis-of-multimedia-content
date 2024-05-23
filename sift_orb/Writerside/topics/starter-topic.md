# About SIFT&ORB

源代码见main.py，项目内容及结果如下。

## 图像匹配

基于以下特征，实现两幅相似图像匹配，提交源码

- SIFT
- ORB

## 帮助

```shell
./sift.py -h
SIFT.

Use SIFT to match two pictures.

usage: sift.py [-hV] [-r <ratio>] [-o <output>] <image1> <image2>

options:
    -h, --help                  Show this screen.
    -V, --version               Show version.
    -n, --dry-run               Without an image shown.
    -d, --debug                 Show log, slow but clear process
    -r, --ratio <ratio>         Lowe ratio.(tolerant:0.6; extreme:0.4)
                                [default: 0.5]
    -o, --output <output>       Output to an image file. 
                                [default: figures/output/sift_t.png]
    
./sift.py -o figures/input0.png figures/input1.png
# A window will be opened with an image file saved.
# Powered by PythonSIFT Open source project
# It takes about 2.835 753.301 seconds in my PC. (Intel i9-13900H)
./sift.py -d -o figures/input0.png figures/input1.png
# Powered by Opencv library functions
# It takes only 2.835 seconds in my PC. (Intel i9-13900H)
./sift.py -no figures/input0.png figures/input1.png
# No window will be opened, but an image file will be saved.
```

## 项目介绍

### 特征描述符求解部分介绍

本程序通过orb算法求解特征描述符，并通过两种方式实现了该部分算法：

1. 基于[开源项目PythonSIFT](https://github.com/rmislam/PythonSIFT)实现了完整的 sift 算法流程(需要在运行参数中加'-d')，
   缺点在于未进行任何优化，运行速度缓慢，但能清晰展现算法原理与实现思路。
2. 直接使用opencv的库函数SIFT完成求解特征描述符, 优点是计算迅速快捷，确定是无法了解SIFT算法原理

### 特征匹配部分介绍

代码基于 ORB 算法得到特征描述符，并使用 FlannBasedMatcher 接口以及函数 FLANN (Fast Library for Approximate Nearest
Neighbors , 最近邻快速搜索库) 对特征描述符进行暴力匹配，根据SIFT作者Lowe’s算法来过滤掉错误的匹配。

Lowe's 算法的工作方式是取一幅图像中的一个 SIFT 关键点，
并找出其与另一幅图像中欧式距离最近的前两个关键点，在这两个关键点中，如果最近的距离除以次近的距离得到的比率 ratio 少于某个阈值，
则接受这一对匹配点。根据资料, Lowe 推荐 ratio 的阈值为0.8, 但经过多次尝试, 结果表明
ratio 取值在0.4~0.6之间最佳, 小于0.4的很少有匹配点, 大于0.6的存在大量错误匹配点。

### 运行结果展示

代码中分别尝试运行了三种模式(加入'-d'参数, 基于[PythonSIFT](https://github.com/rmislam/PythonSIFT)):

1. more-tolerant: ratio = 0.6;
2. default: ratio =0.5;
3. more-extreme: ratio = 0.4

最终结果如下图所示：

<procedure title="生成图像" id="sift">
    <step>
        <p>ratio = 0.4 :</p>
        <img src="match_e.png" alt="more tolerant" border-effect='line'/>
        <p>严格的筛选阈值，可以看到匹配的特征较稀疏，但非常准确</p>
    </step>
    <step>
        <p>ratio = 0.5 :</p>
        <img src="match_d.png" alt="default" border-effect='line'/>
        <p>默认的筛选阈值，特征数量多且准确率高</p>
    </step>
    <step>
        <p>ratio = 0.6 :</p>
        <img src="match_t.png" alt="default" border-effect='line'/>
        <p>宽松的筛选阈值，特征数量非常多但出现了少量错误匹配</p>
    </step>
</procedure>
