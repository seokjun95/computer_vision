import cv2
import numpy as np
import math as m
import os
import time


def padding_tb(img, size):
    imgarr = np.asarray(img, dtype="int32")
    toprow = imgarr[0:size]
    bottomrow = imgarr[-size:]
    imgarr = np.vstack((toprow, imgarr))
    imgarr = np.vstack((imgarr, bottomrow))
    return imgarr


def padding_lr(img, size):
    imgarr = np.asarray(img)
    leftcol = imgarr[:, 0:size]
    rightcol = imgarr[:, -size:]
    leftcol = leftcol.reshape((img.shape[0], size))
    rightcol = rightcol.reshape((img.shape[0], size))
    imgarr = np.hstack((leftcol, imgarr))
    imgarr = np.hstack((imgarr, rightcol))
    return imgarr


def padding_2d(img, r_size, c_size):
    imgarr = np.asarray(img, dtype="int32")
    leftcol = imgarr[:, 0:c_size]
    rightcol = imgarr[:, -c_size:]
    leftcol = leftcol.reshape((img.shape[0], c_size))
    rightcol = rightcol.reshape((img.shape[0], c_size))
    imgarr = np.hstack((leftcol, imgarr))
    imgarr = np.hstack((imgarr, rightcol))
    toprow = imgarr[0:r_size]
    bottomrow = imgarr[-r_size:]
    imgarr = np.vstack((toprow, imgarr))
    imgarr = np.vstack((imgarr, bottomrow))
    return imgarr
# image padded


def cross_correlation_1d(img, kernel):
    if kernel.shape[0] == 1:  # horizon
        size = kernel.shape[1] // 2
        img_arr = padding_lr(img, size)
        filtered_array = np.zeros([img_arr.shape[0], img_arr.shape[1]])
        for i in range(0, img_arr.shape[0]):
            for j in range(size, img_arr.shape[1] - size):
                filtered_array[i][j] = np.sum(kernel * img_arr[i][j - size:j + size + 1])
        filtered_array.astype('uint8')
        ret = filtered_array[0:img_arr.shape[0],size:size+img.shape[1]]
        return ret
    elif kernel.shape[1] == 1:  # vertical
        size = kernel.shape[0] // 2
        img_arr = padding_tb(img, size)
        img_arr = np.transpose(img_arr)
        kernel = np.transpose(kernel)
        filtered_array = np.zeros([img_arr.shape[0], img_arr.shape[1]])
        for i in range(0, img_arr.shape[0]):
            for j in range(size, img_arr.shape[1] - size):
                filtered_array[i][j] = np.sum(kernel * img_arr[i][j - size:j + size + 1])

        filtered_array = np.transpose(filtered_array)
        filtered_array.astype('uint8')
        ret = filtered_array[size:size+img.shape[0],0:img_arr.shape[1]]
        return ret


def cross_correlation_2d(img, kernel):
    v_size = kernel.shape[0] // 2
    h_size = kernel.shape[1] // 2
    arr = padding_lr(img,h_size)
    img_arr = padding_tb(arr,v_size)
    filtered_array = np.zeros([img_arr.shape[0], img_arr.shape[1]])
    for i in range(v_size, img_arr.shape[0] - v_size):
        for j in range(h_size, img_arr.shape[1] - h_size):
            val = np.sum(np.multiply(kernel, img_arr[i-v_size:i+v_size+1,j-v_size:j+v_size+1]))
            filtered_array[i][j] = val
    ret = filtered_array[v_size:v_size+img.shape[0],h_size:h_size+img.shape[1]]
    return ret


def get_gaussian_filter_1d(size, sigma):
    a = np.zeros(size)[np.newaxis]
    mid = size // 2
    for i in range(0, size):
        a[0][i] = (1 / (m.sqrt(2 * m.pi) * sigma)) * (m.exp(-(mid - i) * (mid - i) / (2 * sigma * sigma)))
    a = a/(np.sum(a))
    return a


def get_gaussian_filter_2d(size, sigma):
    r = np.zeros((size,size))
    mid = size // 2
    for i in range(0, size):
        r[0][i] = (m.exp(-(mid - i) * (mid - i) / (2 * sigma * sigma)))
    c = np.transpose(r)
    a = np.asmatrix(r)
    b = np.asmatrix(c)
    ret = (1/(2* m.pi * sigma * sigma))*b*a
    ret = ret/np.sum(ret)
    return ret


def compute_image_gradient(img):
   sob_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
   sob_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
   x_deriv = cross_correlation_2d(img,sob_x)
   y_deriv = cross_correlation_2d(img,sob_y)
   mag = np.sqrt(x_deriv*x_deriv + y_deriv*y_deriv)
   dir = np.arctan2(y_deriv,x_deriv)
   dir = np.rad2deg(dir) + 180
   return mag,dir



def quantize(i):
    if i < 0:
        i += 360
    if 0<= i <22.5 or 337.5 < i <=360:
        i = 0
    elif 22.5<= i < 67.5:
        i = 45
    elif 67.5<= i < 112.5:
        i = 90
    elif 112.5<= i <157.5:
        i = 135
    elif 157.5<= i < 202.5:
        i = 180
    elif 202.5<= i < 247.5:
        i = 225
    elif 247.5<= i < 292.5:
        i = 270
    elif 292.5<= i < 337.5:
        i = 315
    return i


def non_maximum_suppression_dir (mag, dir):
    for i in range(0,dir.shape[0]):
        for j in range(0, dir.shape[1]):
            dir[i][j] = quantize(dir[i][j])
    for i in range(0, dir.shape[0]):
        for j in range(0, dir.shape[1]):
            if dir[i][j] == 0 or dir[i][j] == 180:
                if j-1 < 0:
                    n1 = mag[i][j]
                else:
                    n1 = mag[i][j-1]
                if j+1 >= dir.shape[1]:
                    n2 = mag[i][j]
                else:
                    n2 = mag[i][j+1]
                if mag[i][j] < n1 or mag[i][j] < n2:
                    mag[i][j] = 0

            if dir[i][j] == 45 or dir[i][j] == 225:
                if i-1 < 0 or j+1 >= dir.shape[1]:
                    n1 = mag[i][j]
                else:
                    n1 = mag[i-1][j+1]
                if i+1 >= dir.shape[1] or j-1 < 0:
                    n2 = mag[i][j]
                else:
                    n2 = mag[i+1][j-1]
                if mag[i][j] < n1 or mag[i][j] < n2:
                    mag[i][j] = 0
            if dir[i][j] == 90 or dir[i][j] == 270:
                if i-1 < 0:
                    n1 = mag[i][j]
                else:
                    n1 = mag[i-1][j]
                if i+1 >= dir.shape[0]:
                    n2 = mag[i][j]
                else:
                    n2 = mag[i+1][j]
                if mag[i][j] < n1 or mag[i][j] < n2:
                    mag[i][j] = 0
            if dir[i][j] == 135 or dir[i][j] == 315:
                if j-1 < 0 or i-1 < 0:
                    n1 = mag[i][j]
                else:
                    n1 = mag[i-1][j-1]
                if j+1 >= dir.shape[1] or i+1 >= dir.shape[0]:
                    n2 = mag[i][j]
                else:
                    n2 = mag[i+1][j+1]
                if mag[i][j] < n1 or mag[i][j] < n2:
                    mag[i][j] = 0
    return mag

def compute_coner_response(img):
    kernel = get_gaussian_filter_2d(7, 1.5)
    img = cross_correlation_2d(img, kernel)
    sob_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sob_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    x_deriv = cross_correlation_2d(img, sob_x)
    y_deriv = cross_correlation_2d(img, sob_y)

    R = np.zeros(img.shape)
    x_deriv_d = x_deriv*x_deriv
    y_deriv_d = y_deriv*y_deriv
    xy_d = x_deriv*y_deriv
    for i in range(2, img.shape[0] - 2):
        for j in range(2, img.shape[1] - 2):
            window_x_d = x_deriv_d[i-2:i+3,j-2:j+3]
            window_y_d = y_deriv_d[i-2:i+3,j-2:j+3]
            window_xy = xy_d[i-2:i+3,j-2:j+3]

            rx = window_x_d.sum()
            ry = window_y_d.sum()
            rxy = window_xy.sum()
            R[i][j] = rx*ry-rxy*rxy - 0.04*(rx+ry)*(rx+ry)
            if R[i][j] < 0:
                R[i][j] = 0
    cv2.normalize(R, R, 0.0, 1.0, cv2.NORM_MINMAX)
    return R


def non_maximum_suppression_win(R, winSize):
    for i in range(int(winSize/2), R.shape[0]-int(winSize/2)):
        for j in range(int(winSize/2), R.shape[1]-int(winSize/2)):
            if R[i][j] == np.amax(R[i-int(winSize/2):i+int(winSize/2)+1, j-int(winSize/2):j+int(winSize/2)+1]):
                if R[i][j] <= 0.1:
                    R[i][j] = 0
                else:
                    R[i][j] = 0
    return R



