import numpy as np


def checkerboard(I1, I2, n=12):
    assert I1.shape == I2.shape
    height, width, channels = I1.shape
    hi, wi = height // n, width // n
    outshape = (hi * n, wi * n, channels)

    out_image = np.zeros(outshape, dtype='uint8')
    for i in range(n):
        h = hi * i
        h1 = h + hi
        for j in range(n):
            w = wi * j
            w1 = w + wi
            if (i - j) % 2 == 0:
                out_image[h:h1, w:w1, :] = I1[h:h1, w:w1, :]
            else:
                out_image[h:h1, w:w1, :] = I2[h:h1, w:w1, :]

    return out_image
