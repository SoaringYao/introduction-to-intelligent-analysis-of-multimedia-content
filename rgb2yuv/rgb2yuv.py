"""rgb to yuv.

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
"""

from typing import TYPE_CHECKING

import torch

# static_checking
if TYPE_CHECKING:
    from torch import Tensor


# main class
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


# According to whether the common Y component is divided into three categories :YUV 444,YUV 422,YUV 420
class YUV444(YUV4):

    def __init__(self, yuv4: "Tensor"):
        # Inheriting from a parent class
        super().__init__(yuv4)
        # According to YUV444, expand u v from dimension 0 as well
        self.u = self.u.flatten()
        self.v = self.v.flatten()


class YUV422(YUV4):
    # According to YUV 422,Y shares 2 sets of uv components

    def __init__(self, yuv4: "Tensor"):
        super().__init__(yuv4)
        # down sampling while expanding the color components
        self.u = self.u[0::2, :].flatten()
        # A two-dimensional array is sliced with a step of 2 for rows and 1 for columns
        self.v = self.v[0::2, :].flatten()


class YUV420(YUV4):
    # According to YUV 422,Y shares 4 sets of uv components

    def __init__(self, yuv4: "Tensor"):
        super().__init__(yuv4)
        # down sampling while expanding the color components
        self.u = self.u[0::2, 0::2].flatten()
        # A two-dimensional array is sliced with a step of two for both rows and columns
        self.v = self.v[0::2, 0::2].flatten()


# Twelve YUV formats are implemented according to three broad categories

# YUV444:i444,yv24,nv24,nv42
class I444(YUV444):

    def __init__(self, yuv444: "Tensor"):
        super().__init__(yuv444)

    def __call__(self) -> "Tensor":
        return self.uuvv()


class YV24(YUV444):

    def __init__(self, yuv444: "Tensor"):
        super().__init__(yuv444)

    def __call__(self) -> "Tensor":
        return self.vvuu()


class NV24(YUV444):

    def __init__(self, yuv444: "Tensor"):
        super().__init__(yuv444)

    def __call__(self) -> "Tensor":
        return self.uvuv()


class NV42(YUV444):

    def __init__(self, yuv444: "Tensor"):
        super().__init__(yuv444)

    def __call__(self) -> "Tensor":
        return self.vuvu()


# YUV422:i422,yv16,nv16,nv61
class I422(YUV422):

    def __init__(self, yuv422: "Tensor"):
        super().__init__(yuv422)

    def __call__(self) -> "Tensor":
        return self.uuvv()


class YV16(YUV422):

    def __init__(self, yuv422: "Tensor"):
        super().__init__(yuv422)

    def __call__(self) -> "Tensor":
        return self.vvuu()


class NV16(YUV422):

    def __init__(self, yuv422: "Tensor"):
        super().__init__(yuv422)

    def __call__(self) -> "Tensor":
        return self.uvuv()


class NV61(YUV422):

    def __init__(self, yuv422: "Tensor"):
        super().__init__(yuv422)

    def __call__(self) -> "Tensor":
        return self.vuvu()


# YUV420:i420,yv12,nv12,nv21
class I420(YUV420):

    def __init__(self, yuv420: "Tensor"):
        super().__init__(yuv420)

    def __call__(self) -> "Tensor":
        return self.uuvv()


class YV12(YUV420):

    def __init__(self, yuv420: "Tensor"):
        super().__init__(yuv420)

    def __call__(self) -> "Tensor":
        return self.vvuu()


class NV12(YUV420):
    def __init__(self, yuv420: "Tensor"):
        super().__init__(yuv420)

    def __call__(self) -> "Tensor":
        return self.uvuv()


class NV21(YUV420):

    def __init__(self, yuv420: "Tensor"):
        super().__init__(yuv420)

    def __call__(self) -> "Tensor":
        return self.vuvu()


# Dictionary definition
YUV_DICT = {
    "i444": I444,
    "yv24": YV24,
    "nv24": NV24,
    "nv42": NV42,
    "i422": I422,
    "yv16": YV16,
    "nv16": NV16,
    "nv61": NV61,
    "i420": I420,
    "yv12": YV12,
    "nv12": NV12,
    "nv21": NV21,
}


# The color space is converted from RGB to YUV
def rgb2yuv(inp: "Tensor") -> "Tensor":
    bt = torch.tensor([[0.299, 0.587, 0.114],
                       [-0.148, -0.289, 0.437],
                       [0.615, -0.515, -0.100], ])
    out = inp @ bt.transpose(1, 0) + torch.tensor([0, 128, 128])
    return out


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


# main function
if __name__ == "__main__" and __doc__:
    from docopt import docopt

    args = docopt(__doc__)
    if args["--debug"]:
        print(args)

    rgb = torch.from_numpy(read_bmp(args["<inp>"]).copy()) + 0.0

    # Not used in real programs, only to verify that the rgb read is correct
    # {{ verifying
    from matplotlib.image import imread

    rgb_verify = torch.from_numpy(imread(args["<inp>"]).copy()) + 0.0
    if (rgb == rgb_verify).all():
        print('\nrgb parses correctly\n')
    else:
        exit(-1)
    # }} verifying end

    if args["--verbose"]:
        image_size = "*".join(list(map(str, rgb.shape[0:2][::-1])))
        print(f"width*height is {image_size}.")
    yuv = rgb2yuv(rgb)
    yuv = yuv.int().maximum(torch.tensor(0)).minimum(torch.tensor(255))
    if args["--debug"]:
        print(yuv[:, :, 1])
        print(yuv[:, :, 2])
    yuv_file = YUV_DICT[args["--format"]](yuv)()
    with open(args["<out>"], mode="wb") as f:
        f.write(bytes(yuv_file.tolist()))
