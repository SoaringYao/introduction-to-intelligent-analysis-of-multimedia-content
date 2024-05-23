# 多媒体内容智能分析导论

## 实现图像读取和颜色空间转换

- 将文件夹中的BMP图像读取，然后将RGB图像转换到YUV颜色空间并保存
- 不能调用现有的图像读取函数、颜色空间转换函数，代码自己编写

## 程序介绍

本程序主要通过pytorch完成，利用了pytorch对张量的强大处理能力实现了rgb到yuv的颜色转换以及不同yuv格式的转换，
此外，通过面向对象的编程方式实现了对12种不同格式的yuv的读取和处理。

## help

```shell
# change rgb to yuv.

usage: rgb2yuv.py [-hdv] [-f <fmt>] <inp> <out>

options:
    -h, --help
    -d, --debug             print yuv tensor information.
    -v, --verbose           print information about image.
    -f, --format <fmt>      Format of yuv. [default: i420]

formats:
    444:
        i444: YYY UUU VVV
        yv24: YYY VVV UUU
        nv24: YYY UV UV UV
        nv42: YYY VU VU VU
    422:
        i422: YYYY UU VV
        yv16: YYYY VV UU
        nv16: YYYY UV UV
        nv61: YYYY VU VU
    420:
        i420: YYYYYYYY UU VV
        yv12: YYYYYYYY UU VV
        nv12: YYYYYYYY UV UV
        nv21: YYYYYYYY VU VU
```

## 注意事项

- 通过 [YUView](https://ient.github.io/YUView) 查看 yuv 格式文件验证了实验结果。

- yuv 文件不含宽高信息，需要在 YUView 中手动输入，可以通过在运行程序时添加参数-v打印图像宽高信息

- YUView 中默认格式包括 YUV 4:2:0 8-bit、YUV 4:2:2 8-bit、YUV 4:4:4 8-bit，
  分别对应代码中的i420、i422、i444三种格式。其他格式无法通过该软件查看。

- 程序默认生成的 yuv 的格式为i420，可以通过-f参数修改为其他格式，代码通过docopt包定义接口描述，具体参数和使用方法已在help部分给出。

- .run文件夹中有设置好的pycharm运行配置，rgb2yuv_default以参数-d -v tests/rgb.bmp tests/yuv.yuv运行
  输入输出文件路径为tests/rgb.bmp和tests/yuv.yuv。

## 部分关键代码和注释

```python
def read_bmp(filename):
    bmp_header_size = 54

    import numpy as np

    # read bmp files
    with open(filename, 'rb') as fr:
        # read the bmp file header
        bmp_header = fr.read(bmp_header_size)

        # get the image width and height
        width = bmp_header[18] + (bmp_header[19] << 8) + (bmp_header[20] << 16) + (bmp_header[21] << 24)
        height = bmp_header[22] + (bmp_header[23] << 8) + (bmp_header[24] << 16) + (bmp_header[25] << 24)

        # reading pixel data
        pixel_data = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height - 1, -1, -1):
            for j in range(width):
                # the b g and r components of the pixel are read
                b = ord(fr.read(1))
                g = ord(fr.read(1))
                r = ord(fr.read(1))
                # the storage order is rgb
                pixel_data[i, j] = [r, g, b]
                # skip the placeholder bytes
                fr.read(1)

    return pixel_data
```

read_bmp程序用于读取bmp文件，并将其转换为numpy数组，在该部分需要注意
每一位像素的颜色空间按b、g、r存储，并且额外补一位255作为占位符，因此需要跳过该位。
此外bmp文件按从左到右、从下到上存储，因此生成numpy数组时需要注意存储顺序，对行遍历时应从高向低。

```python
# Not used in real programs, only to verify that the rgb read is correct
# {{ verifying
from matplotlib.image import imread

rgb_verify = torch.from_numpy(imread(args["<inp>"]).copy()) + 0.0
if (rgb == rgb_verify).all():
    print('\nrgb parses correctly\n')
else:
    exit(-1)
# }} verifying end
```

此部分代码使用了matplotlib.image库，但该部分并未实际用于rgb颜色空间的读取，
仅作为调试用途存在，用于校验read_bmp读取的rgb颜色空间是否正确，

```python
class YUV4:

    def __init__(self, inp: "Tensor"):
        # Dimension transformation, which separates y,u,v into three two-dimensional matrices
        self.y, self.u, self.v = inp.permute(2, 0, 1)
        # expand the matrix from dimension 0
        self.y = self.y.flatten()

    def uuvv(self) -> "Tensor":
        # concatenate tensors row by row
        uuvv = torch.cat([self.y, self.u, self.v])
        return uuvv

    def vvuu(self) -> "Tensor":
        vvuu = torch.cat([self.y, self.v, self.u])
        return vvuu

    def uvuv(self) -> "Tensor":
        # y is stored separately and u v is stored alternately by two tensor concatenations
        uv = torch.stack([self.u, self.v], dim=1).flatten()
        # stack the u,v tensors in rows
        uvuv = torch.cat([self.y, uv])
        # concatenate tensors row by row
        return uvuv

    def vuvu(self) -> "Tensor":
        uv = torch.stack([self.v, self.u], dim=1).flatten()
        vuvu = torch.cat([self.y, uv])
        return vuvu


class YUV420(YUV4):
    # According to YUV 422,Y shares 4 sets of uv components

    def __init__(self, yuv4: "Tensor"):
        super().__init__(yuv4)
        # down sampling while expanding the color components
        self.u = self.u[0::2, 0::2].flatten()
        # A two-dimensional array is sliced with a step of two for both rows and columns
        self.v = self.v[0::2, 0::2].flatten()


class I420(YUV420):

    def __init__(self, yuv420: "Tensor"):
        super().__init__(yuv420)

    def __call__(self) -> "Tensor":
        return self.uuvv()
```

上面代码节选了i420格式相关类的定义。完整的12种YUV格式类定义请参考源代码。

在实现i420格式时，首先定义了YUV4类，该类定义了YUV4格式中所有uv信息的存储方式，并提供了uv信息的拼接方法。
其次定义了YUV420类，该类继承了YUV4类，根据YUV420的具体要求重写了uv信息的拼接方法，实现了uv信息的下采样，做到4个Y分量共用一组uv分量。
最后定义了I420类，该类继承了YUV420类，根据I420的具体要求重写了uv信息的拼接方法，实现了先存储U分量再存储V分量。

对于其他的格式，使用了torch.cat()和torch.stack()方法，用不同方式拼接存储U、V分量的矩阵，
并利用torch.flatten()方法将拼接后的矩阵展开，实现了uv信息的拼接与存储。

## 结果展示

下图为输出的yuv.yuv文件通过YUView查看得到的画面
![image](result.png)

## 总结

本程序实现了读取bmp文件，将其从rgb颜色空间转换为YUV颜色空间并进行存储的功能。
程序利用面向对象的方法实现了12种YUV格式的转换和存储，并且提供了各类参数方便调试与使用。
