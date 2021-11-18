from crc_mapping import camera_response_curve, hdr_gen, tone_mapping
import os
import time
import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt


def load_images_and_exposure_times(path):
    print("-----------------------------------------------------")
    print("-- images and exposure times loading")
    file_name = []
    exposure_time_list = []
    src_img_list = []
    t = time.time()
    for files in os.listdir(path):
        if "burst.csv" == files:
            f = open(os.path.join(path, files), 'r')
            lines = f.readlines()
            for i in range(1, len(lines)):
                line = lines[i]
                (file, exposure, *rest) = line.split(",")
                if (".jpg" in file) or (".png" in file):
                    file_name.append(file)
                    exposure_time_list.append(np.float32(exposure))

    for f in file_name:
        img = cv2.imread(os.path.join(path, f))
        src_img_list.append(img)
    duration = round(time.time() - t, 3)
    print("-- reading image finish in " + str(duration) + "s")
    return src_img_list, exposure_time_list


def camera_curve_plot(g_r, g_g, g_b):
    plt.figure(figsize=(10, 10))
    plt.plot(g_r, range(256), 'rx')
    plt.plot(g_g, range(256), 'gx')
    plt.plot(g_b, range(256), 'bx')
    plt.ylabel('pixel value Z')
    plt.xlabel('log exposure X')
    plt.savefig('response-curve.png')


def save_radiance_map(hdr):
    plt.figure(figsize=(12, 8))
    plt.imshow(np.log(cv2.cvtColor(hdr, cv2.COLOR_BGR2GRAY)), cmap='jet')
    plt.colorbar()
    plt.savefig('radiance-map.png')


if __name__ == "__main__":
    src_path = "./images"
    src_img_list, exposure_time_list = load_images_and_exposure_times(src_path)

    image_list_r = []
    image_list_g = []
    image_list_b = []

    for img in src_img_list:
        image_list_r.append(img[:, :, 0])
        image_list_g.append(img[:, :, 1])
        image_list_b.append(img[:, :, 2])

    print("-----------------------------------------------------")
    print("-- camera response curve generation")
    t = time.time()
    g_r, _ = camera_response_curve(image_list_r, exposure_time_list)
    g_g, _ = camera_response_curve(image_list_g, exposure_time_list)
    g_b, _ = camera_response_curve(image_list_b, exposure_time_list)
    camera_curve_plot(g_r, g_g, g_b)
    print("-- camera response curve generation in " + str(round(time.time() - t, 3)) + "s")

    print("-----------------------------------------------------")
    print("-- hdr generation")
    t = time.time()
    hdr = hdr_gen([image_list_r, image_list_g, image_list_b], exposure_time_list, [g_r, g_g, g_b])
    save_radiance_map(hdr)
    cv2.imwrite("hdr.png", hdr)
    print("-- hdr generation in " + str(round(time.time() - t, 3)) + "s")
    print("-----------------------------------------------------")

    print("-----------------------------------------------------")
    print("-- tone mapping")
    t = time.time()
    ldr = tone_mapping(hdr)
    cv2.imwrite("ldr.png", ldr)
    print("-- tone mapping in " + str(round(time.time() - t, 3)) + "s")
    print("-----------------------------------------------------")