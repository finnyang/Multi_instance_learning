import cv2
import numpy as np


def dark_channel(image, size=15):
    dark_ = image.min(axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark_ = cv2.erode(dark_, kernel)
    return dark_


def calculate_A(image, dc):
    size = image.shape[0:2]
    length = size[0]*size[1]
    top_k = int(length/1000.0)
    image = image.reshape((-1, 3))
    dc = dc.reshape((-1))
    index = dc.argsort()[::-1][0:top_k+1]
    top_k_data = image[index, :]
    A = top_k_data.mean(axis=0)
    return A


def calculate_t(image, A):
    image = image/A
    t = 1 - 0.95*dark_channel(image)
    return t


def calculate_J(image, A, t):
    t = t.clip(0.1, 1)
    t = t.reshape((t.shape[0], t.shape[1], 1))
    t = np.concatenate([t, t, t], 2)
    J = (image-A)/t+A
    return J


def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im*p, cv2.CV_64F, (r,r))
    cov_Ip = mean_Ip - mean_I*mean_p

    mean_II = cv2.boxFilter(im*im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I*mean_I

    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a*im + mean_b
    return q


def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray, et, r, eps)
    return t


def main():
    image_path = "2.jpeg"
    image = cv2.imread(image_path)
    image_float = image.copy().astype(np.float32)/255.0
    dc = dark_channel(image_float)
    A = calculate_A(image_float, dc)
    t = calculate_t(image_float, A)
    t = TransmissionRefine(image, t)
    J = calculate_J(image_float, A, t)
    J = J*255
    cv2.imwrite("haze_2.jpg", J)


if __name__ == "__main__":
    main()
