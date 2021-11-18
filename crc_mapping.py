import math

import cv2
import numpy as np

PIXEL_MIN = 0
PIXEL_MAX = 255


def camera_response_curve(image_list, exposure_times, number_of_samples_per_dimension=20):
    """
    @description: 相机响应曲线恢复
    @params:
        image_list: 序列图像列表, channel = 1
        exposure_time: 曝光时间列表
        number_of_samples_per_dimension: 每张图像水平和垂直方向取样数量
    @return:
        g: 相机响应曲线(离散列表)
        ln_e: 响应曲线偏移(常量)
    """
    if len(image_list) != len(exposure_times):
        print("-- error: image_list length not equal to exposure time list.")
        return

    if len(image_list[0].shape) > 2:
        print("-- error: image must be single channel.")
        return

    width = image_list[0].shape[0]
    height = image_list[0].shape[1]
    width_iter = width / number_of_samples_per_dimension
    height_iter = height / number_of_samples_per_dimension

    # 取样
    Z = np.zeros((len(image_list), number_of_samples_per_dimension * number_of_samples_per_dimension))

    for index, image in enumerate(image_list):
        h_iter = 0
        for i in range(number_of_samples_per_dimension):
            w_iter = 0
            for j in range(number_of_samples_per_dimension):
                if math.floor(w_iter) < width and math.floor(h_iter) < height:
                    pixel = image[math.floor(w_iter), math.floor(h_iter)]
                    Z[index, i * number_of_samples_per_dimension + j] = pixel
                w_iter += width_iter
            h_iter += height_iter

    B = [np.log(exposure) for exposure in exposure_times]
    l = 50
    w = [z if z < 0.5 * PIXEL_MAX else PIXEL_MAX - z for z in range(PIXEL_MAX + 1)]

    # 计算方程3
    n = PIXEL_MAX + 1
    A = np.zeros(shape=(np.size(Z, 0) * np.size(Z, 1) + n + 1, n + 1 + np.size(Z, 1)), dtype=np.float32)
    b = np.zeros(shape=(np.size(A, 0), 1), dtype=np.float32)

    k = 0
    for i in range(np.size(Z, 1)):
        for j in range(np.size(Z, 0)):
            z = int(Z[j][i])
            wij = w[z]
            A[k][z] = wij
            A[k][n + i] = -wij
            b[k] = wij * B[j]
            k += 1

    A[k][n // 2] = 1
    k += 1

    # smooth
    for i in range(n - 1):
        A[k][i] = l * w[i + 1]
        A[k][i + 1] = -2 * l * w[i + 1]
        A[k][i + 2] = l * w[i + 1]
        k += 1

    x = np.linalg.lstsq(A, b, rcond=-1)[0]
    g = x[:n]
    ln_e = x[n:]

    return g, ln_e


def hdr_gen(image_list, exposure_times, camera_response_curve):
    """
    @description: 根据相机响应曲线和曝光时间生成HDR图像
    @params:
        image_list: 序列图像列表, length = 3
        exposure_time: 曝光时间列表
        camera_response_curve: 相机响应曲线, length = 3 (RGB)
    @return:
        hdr: 生成的HDR图像
    """

    img_size = image_list[0][0].shape
    w = [z if z <= 0.5 * PIXEL_MAX else PIXEL_MAX-z for z in range(PIXEL_MAX + 1)]
    ln_t = np.log(exposure_times)

    hdr = np.zeros(shape=(img_size[0], img_size[1], 3), dtype=np.float32)
    vfunc = np.vectorize(lambda x: math.exp(x))

    for i in range(3):
        print(' - Constructing radiance map for {0} channel .... '.format('RGB'[i]))
        Z = [img.flatten().tolist() for img in image_list[i]]
        E = radiance_map_gen(camera_response_curve[i], Z, ln_t, w)
        hdr[..., i] = np.reshape(vfunc(E), img_size)

    print(np.max(hdr))
    return hdr


def radiance_map_gen(g, Z, ln_t, w):
    """
    @description: 生成辐照图，论文公式6
    @params:
        g: 响应曲线
        Z: 图像像素
        ln_t: 曝光时间
        w: 权重
    @return:
        ln_E: 恢复出来的辐照图
    """
    acc_E = [0.0] * len(Z[0])
    ln_E = [0.0] * len(Z[0])

    pixels, imgs = len(Z[0]), len(Z)
    for i in range(pixels):
        acc_w = 0
        for j in range(imgs):
            z = Z[j][i]
            acc_E[i] += w[z] * (g[z] - ln_t[j])
            acc_w += w[z]
        ln_E[i] = acc_E[i] / acc_w if acc_w > 0 else acc_E[i]

    return ln_E


def tone_mapping(hdr):
    """
    @description: 色调映射，将HDR转换为LDR
    @param:
        hdr: HDR图像
    @return:
        ldr: 色调映射后的LDR图
    """
    #tone_map = cv2.createTonemapReinhard(5, 0, 0.5, 0.5)
    tone_map = cv2.createTonemapDrago(3)
    ldr = tone_map.process(hdr)

    return ldr * 255
