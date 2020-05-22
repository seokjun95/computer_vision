import cv2 as cv
import numpy as np
import math

smile = cv.imread('smile.png', cv.IMREAD_GRAYSCALE)


def get_transformed_image(img, m):
    x = img.shape[0]  # 50
    y = img.shape[1]  # 55 -> rotate size

    xy = np.asarray(([x, y]))
    xy = np.matmul(m, xy)
    xy = np.rint(xy)
    xy = xy.astype(int)
    invm = np.linalg.inv(m)

    if m[0][0] != 1 and m[1][1] != 1 and m[0][1] != 0:
        if x >= y:
            dia = math.ceil(x*1.08)
        elif y>x:
            dia = math.ceil(y*1.08)

        r_frame = np.full((dia+1, dia+1), -1)
        ret = np.full((dia+1,dia+1),255)
        dia = dia // 2
        for i in range(dia - (x // 2), dia + (x // 2)):
            for j in range(dia - (y // 2), dia + (y // 2)):
                r_frame[i][j] = img[i - dia + (x // 2)][j - dia + (y // 2)]

        for i in range(dia-(x//2), dia+(x//2)):
            for j in range(dia-(y//2), dia+(y//2)):
                ij = np.asarray(([i-dia, j-dia]))
                ij = np.matmul(m, ij)
                ij = ij.astype(int)
                ret[ij[0]+dia][ij[1]+dia] = r_frame[i][j]
        iidx = 0
        jidx = 0
        for i in range(0,ret.shape[0]):
            for j in range(0,ret.shape[1]):
                if ret[i][j] != 255:
                    iidx = i
                    break;
            if iidx != 0:
                break;
        for i in range(0,ret.shape[0]):
            for j in range(0,ret.shape[1]):
                if ret[j][i] != 255:
                    jidx = i
                    break;
            if jidx != 0:
                break;

        for i in range(0,iidx-3):
           ret = np.delete(ret,0,0)
        for i in range(0, iidx-3):
            ret = np.delete(ret, ret.shape[0]-1, 0)
        for i in range(0, jidx - 3):
            ret = np.delete(ret, 0, 1)
        for i in range(0, jidx - 3):
            ret = np.delete(ret, ret.shape[1]-1 , 1)
      
        return ret

    ret_img = np.full((xy[0], xy[1]), 255)
    x2 = xy[0]
    y2 = xy[1]
    for i in range(0, x2):
        for j in range(0, y2):
            ij = np.asarray(([i, j]))
            ij = np.matmul(invm, ij)
            ij = ij.astype(int)
            ret_img[i][j] = img[ij[0]][ij[1]]
    return ret_img


def draw_arrow(img2, m1, m2, r, c):
    plane = np.full((801, 801), 255, np.uint8)
    ar = cv.arrowedLine(plane, (0, 400), (800, 400), 0, thickness=2, tipLength=0.05)
    ar = cv.arrowedLine(plane, (400, 800), (400, 0), 0, thickness=2, tipLength=0.05)
    ararr = np.asarray(ar, dtype="int32")
    ret = ararr
    for i in range(m1 - r, m1 + r):
        for j in range(m2 - c, m2 + c):
            if i > 800:
                i = i - 800
            if j > 800:
                j = j - 800
            if img2[i][j] != 255:
                ret[i][j] = img2[i][j]

    return ret


def img_display(img):
    imgarr = np.asarray(img, dtype="int32")  # 350~450,345~455
    imgarr2 = np.full((801, 801), 255)
    ccr = 0
    cr = 0
    row = imgarr.shape[0] // 2  # 50
    col = imgarr.shape[1] // 2  # 55
    mid1 = 400
    mid2 = 400
    for i in range(mid1 - row, mid1 + row):
        for j in range(mid2 - col, mid2 + col):
            imgarr2[i][j] = imgarr[i - mid1 + row][j - mid2 + col]

    original = imgarr2
    newmid1 = mid1
    newmid2 = mid2
    newrow = row
    newcol = col
    out = draw_arrow(imgarr2, mid1, mid2, row, col)
    cv.imshow('smile.png', out / 255)
    while True:
        shiftedarr = np.full((801, 801), 255)
        key = cv.waitKey(0)
        if key == ord('a'):
            newmid1 = newmid1
            newmid2 = newmid2 - 5
            for i in range(newmid1 - newrow, newmid1 + newrow):
                for j in range(newmid2 - newcol, newmid2 + newcol):
                    shiftedarr[i][j] = imgarr[i - newmid1 + newrow][j - newmid2 + newcol]
            imgarr2 = shiftedarr
            out = draw_arrow(shiftedarr, newmid1, newmid2, row, col)
            cv.imshow('smile.png', out / 255)

        elif chr(key & 255) == 'd':
            newmid1 = newmid1
            newmid2 = newmid2 + 5
            for i in range(newmid1 - row, newmid1 + row):
                for j in range(newmid2 - col, newmid2 + col):
                    if j > 800 :
                        shiftedarr[i][j - 800] = imgarr[i - newmid1 + row][j - newmid2 + col]
                    else:
                        shiftedarr[i][j] = imgarr[i - newmid1 + row][j - newmid2 + col]
            imgarr2 = shiftedarr
            out = draw_arrow(shiftedarr, newmid1, newmid2, row, col)
            cv.imshow('smile.png', out / 255)

        elif chr(key & 255) == 'w':
            newmid1 = newmid1 - 5
            newmid2 = newmid2
            for i in range(newmid1 - row, newmid1 + row):
                for j in range(newmid2 - col, newmid2 + col):
                    shiftedarr[i][j] = imgarr[i - newmid1 + row][j - newmid2 + col]
            imgarr2 = shiftedarr
            out = draw_arrow(shiftedarr, newmid1, newmid2, row, col)
            cv.imshow('smile.png', out / 255)

        elif chr(key & 255) == 's':
            newmid1 = newmid1 + 5
            newmid2 = newmid2
            for i in range(newmid1 - row, newmid1 + row):
                for j in range(newmid2 - col, newmid2 + col):
                    if i > 800:
                        shiftedarr[i - 800][j] = imgarr[i - newmid1 + row][j - newmid2 + col]
                    else:
                        shiftedarr[i][j] = imgarr[i - newmid1 + row][j - newmid2 + col]
            imgarr2 = shiftedarr
            out = draw_arrow(shiftedarr, newmid1, newmid2, row, col)
            cv.imshow('smile.png', out / 255)

        elif chr(key & 255) == 'r':
            prevrow = newrow
            xyarr = np.asarray(([math.cos(math.pi/36), -math.sin(math.pi/36)], [math.sin(math.pi/36), math.cos(math.pi/36)]))
           # cr = cr+1
           # xyarr= np.linalg.matrix_power(xyarr,cr)
            new_img = get_transformed_image(imgarr, xyarr)

            newrow = new_img.shape[0] // 2
            newcol = new_img.shape[1] // 2

            m = np.asarray(([newmid1-400, newmid2-400]))
            m = np.matmul(xyarr, m)
            m = m.astype(int)
            newmid1 = m[0]+400
            newmid2 = m[1]+400

            for i in range(newmid1 - newrow, newmid1 + newrow):
                for j in range(newmid2 - newcol, newmid2 + newcol):
                    shiftedarr[i][j] = new_img[i - newmid1 + newrow][j - newmid2 + newcol]
            imgarr2 = shiftedarr
            col = newcol
            row = newrow
            out = draw_arrow(shiftedarr, newmid1, newmid2, newrow, newcol)
            cv.imshow('smile.png', out / 255)
            imgarr = new_img
        elif chr(key & 255) == 'R':
            prevrow = newrow
            xyarr = np.asarray(
                ([math.cos(-math.pi / 36), -math.sin(-math.pi / 36)], [math.sin(-math.pi / 36), math.cos(-math.pi / 36)]))
            # cr = cr+1
            # xyarr= np.linalg.matrix_power(xyarr,cr)
            new_img = get_transformed_image(imgarr, xyarr)

            newrow = new_img.shape[0] // 2
            newcol = new_img.shape[1] // 2

            m = np.asarray(([newmid1 - 400, newmid2 - 400]))
            m = np.matmul(xyarr, m)
            m = m.astype(int)
            newmid1 = m[0] + 400
            newmid2 = m[1] + 400

            for i in range(newmid1 - newrow, newmid1 + newrow):
                for j in range(newmid2 - newcol, newmid2 + newcol):
                    shiftedarr[i][j] = new_img[i - newmid1 + newrow][j - newmid2 + newcol]
            imgarr2 = shiftedarr
            col = newcol
            row = newrow
            out = draw_arrow(shiftedarr, newmid1, newmid2, newrow, newcol)
            cv.imshow('smile.png', out / 255)
            imgarr = new_img
        elif chr(key & 255) == 'f':
            newmid1 = mid1 + (mid1 - newmid1)
            for i in range(newmid1 - row, newmid1 + row):
                for j in range(newmid2 - col, newmid2 + col):
                    shiftedarr[i][j] = imgarr[i - newmid1 + row][j - newmid2 + col]
            imgarr2 = shiftedarr
            out = draw_arrow(shiftedarr, newmid1, newmid2, row, col)
            cv.imshow('smile.png', out / 255)

        elif chr(key & 255) == 'F':
            newmid2 = mid2 + (mid2 - newmid2)
            for i in range(newmid1 - row, newmid1 + row):
                for j in range(newmid2 - col, newmid2 + col):
                    shiftedarr[i][j] = imgarr[i - newmid1 + row][j - newmid2 + col]
            imgarr2 = shiftedarr
            out = draw_arrow(shiftedarr, newmid1, newmid2, row, col)
            cv.imshow('smile.png', out / 255)

        elif chr(key & 255) == 'x':  # 101->96 111->106
            prevrow = newrow
            xyarr = np.asarray(([0.95, 0], [0, 1]))
            new_img = get_transformed_image(imgarr, xyarr)
            imgarr = new_img
            newrow = new_img.shape[0] // 2
            newcol = new_img.shape[1] // 2
            for i in range(newmid1 - newrow, newmid1 + newrow):
                for j in range(newmid2 - newcol, newmid2 + newcol):
                    shiftedarr[i][j] = new_img[i - newmid1 + newrow][j - newmid2 + newcol]

            imgarr2 = shiftedarr
            col = newcol
            row = newrow
            out = draw_arrow(shiftedarr, newmid1, newmid2, newrow, newcol)
            cv.imshow('smile.png', out / 255)

        elif chr(key & 255) == 'X':
            prevrow = newrow
            xyarr = np.asarray(([1.05, 0], [0, 1]))
            new_img = get_transformed_image(imgarr, xyarr)
            imgarr = new_img
            newrow = new_img.shape[0] // 2
            newcol = new_img.shape[1] // 2
            for i in range(newmid1 - newrow, newmid1 + newrow):
                for j in range(newmid2 - newcol, newmid2 + newcol):
                    shiftedarr[i][j] = new_img[i - newmid1 + newrow][j - newmid2 + newcol]

            imgarr2 = shiftedarr
            col = newcol
            row = newrow
            out = draw_arrow(shiftedarr, newmid1, newmid2, newrow, newcol)
            cv.imshow('smile.png', out / 255)

        elif chr(key & 255) == 'y':
            prevrow = newrow
            xyarr = np.asarray(([1, 0], [0, 0.95]))
            new_img = get_transformed_image(imgarr, xyarr)
            imgarr = new_img
            newrow = new_img.shape[0] // 2
            newcol = new_img.shape[1] // 2
            for i in range(newmid1 - newrow, newmid1 + newrow):
                for j in range(newmid2 - newcol, newmid2 + newcol):
                    shiftedarr[i][j] = new_img[i - newmid1 + newrow][j - newmid2 + newcol]

            imgarr2 = shiftedarr
            col = newcol
            row = newrow
            out = draw_arrow(shiftedarr, newmid1, newmid2, newrow, newcol)
            cv.imshow('smile.png', out / 255)

        elif chr(key & 255) == 'Y':
            prevrow = newrow
            xyarr = np.asarray(([1, 0], [0, 1.05]))
            new_img = get_transformed_image(imgarr, xyarr)
            imgarr = new_img
            newrow = new_img.shape[0] // 2
            newcol = new_img.shape[1] // 2
            for i in range(newmid1 - newrow, newmid1 + newrow):
                for j in range(newmid2 - newcol, newmid2 + newcol):
                    shiftedarr[i][j] = new_img[i - newmid1 + newrow][j - newmid2 + newcol]

            imgarr2 = shiftedarr
            col = newcol
            row = newrow
            out = draw_arrow(shiftedarr, newmid1, newmid2, newrow, newcol)
            cv.imshow('smile.png', out / 255)

        elif chr(key & 255) == 'H':
            imgarr2 = original
            imgarr = np.asarray(img, dtype="int32")
            newmid1 = mid1
            newmid2 = mid2
            cr = 0
            ccr = 0
            row = imgarr.shape[0]//2
            col = imgarr.shape[1]//2
            out = draw_arrow(imgarr2, 400, 400, 50, 55)
            cv.imshow('smile.png', out / 255)

        elif chr(key & 255) == 'Q':
            cv.destroyWindow('smile.png')
            break;


img_display(smile)
