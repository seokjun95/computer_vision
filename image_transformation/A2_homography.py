import cv2 as cv
import numpy as np
import math
import random as rd
import time


def compute_homography(srcP, destP):
    sP = srcP
    dP = destP
    sx_m = np.mean(sP, axis=0)[0]
    sy_m = np.mean(sP, axis=0)[1]
    dx_m = np.mean(dP, axis=0)[0]
    dy_m = np.mean(dP, axis=0)[1]
    t_s = np.asarray(([1,0,-sx_m],[0,1,-sy_m],[0,0,1]))
    t_d = np.asarray(([1,0,-dx_m],[0,1,-dy_m],[0,0,1]))
    distance_s = 0
    distance_d = 0
    for i in range(0,sP.shape[0]):
        ds = sP[i][0]**2+sP[i][1]**2
        if ds>distance_s:
            distance_s = ds
        dd = dP[i][0]**2+dP[i][1]**2
        if dd > distance_d:
            distance_d = dd
    s_longest = (2**(1/2))/(distance_s**(1/2))
    d_longest = (2**(1/2))/(distance_d**(1/2))
    s_s = np.asarray(([s_longest,0,0],[0,s_longest,0],[0,0,1]))
    s_d = np.asarray(([d_longest,0,0],[0,d_longest,0],[0,0,1]))
    t_s = np.matmul(s_s,t_s)
    t_d = np.matmul(s_d,t_d)

    sP = np.transpose(sP)
    dP = np.transpose(dP)
    sP = np.matmul(t_s, sP)
    dP = np.matmul(t_d, dP)
    sP = np.transpose(sP)
    dP = np.transpose(dP)

    a = np.array(([]))
    a = a.reshape(0,9)
    for i in range(0,len(srcP)):
        ai = np.asarray(([-sP[i][0],-sP[i][1],-1,0,0,0,(sP[i][0]*dP[i][0]),(sP[i][1]*dP[i][0]),dP[i][0]],[0,0,0,-sP[i][0],-sP[i][1],-1,sP[i][0]*dP[i][1],sP[i][1]*dP[i][1],dP[i][1]]))
        a = np.concatenate((a,ai),axis=0)

    u,s,v = np.linalg.svd(a,full_matrices=True)
    h = v[8, :]
    h = np.reshape(h,(3,3))

    h = np.matmul(np.linalg.inv(t_d), h)
    h = np.matmul(h, t_s)

    return h


def compute_homography_ransac(srcP,destP,th):
    sP = np.full((5, 3),1)
    dP = np.full((5, 3),1)
    inlier = np.full((1,500),0)

    rhs = []
    for k in range(0,500):
        chosen = rd.sample(range(0,100), 5)
        sP[0] = srcP[0]
        dP[0] = destP[0]
        for i in range (1,5):
            sP[i] = srcP[chosen[i]]
            dP[i] = destP[chosen[i]]
        r_h = compute_homography(sP,dP)
        rhs.append(r_h)
        for i in range(0,destP.shape[0]):
            d = np.matmul(r_h,np.transpose(srcP[i]))
            d = np.transpose(d)
            d_i = np.linalg.norm(srcP[i]-d)
            if d_i < th:
                inlier[0][k] += 1

    return rhs,inlier


desk = cv.imread('cv_desk.png', cv.IMREAD_GRAYSCALE)
cover = cv.imread('cv_cover.jpg', cv.IMREAD_GRAYSCALE)
hp = cv.imread('hp_cover.jpg', cv.IMREAD_GRAYSCALE)

hp = cv.resize(hp,dsize=(350,440),interpolation=cv.INTER_AREA)
out = None
orb = cv.ORB_create()
kp1, des1 = orb.detectAndCompute(desk, None)
kp2, des2 = orb.detectAndCompute(cover, None)

tmp = 0
tIdx = 0
qIdx = 0
matches = []
for i in range(0,len(kp1)):
    for j in range(0,len(kp2)):
        tmp = cv.norm(des1[i],des2[j],cv.NORM_HAMMING)
        if j == 0:
            dist = tmp
        elif dist > tmp:
            dist = tmp
            tIdx = i
            qIdx = j
    match = cv.DMatch(tIdx,qIdx,dist)
    matches.append(match)

listtodel = []
for i in range(0,len(matches)):
    x = matches[i].trainIdx
    y = matches[i].queryIdx
    d = matches[i].distance

    for j in range(0,len(matches)):
        if i == j:
            continue
        if matches[i].trainIdx == matches[j].trainIdx and matches[i].queryIdx != matches[j].queryIdx:
            if matches[j].distance > d:
                listtodel.append(j)
        if matches[i].queryIdx == matches[j].queryIdx and matches[i].trainIdx != matches[j].trainIdx:
            if matches[j].distance > d:
                listtodel.append(j)

listtodel = list(set(listtodel))
listtodel.sort(reverse=True)
for i in listtodel:
    del matches[i]

matches = sorted(matches, key=lambda x: x.distance)
out = cv.drawMatches(desk,kp1,cover,kp2,matches[:10],out,flags=2) #feature mathching
cv.imshow("feature_matching.png",out)
#cv.waitKey(0)
N = 15
srcP = np.full((N,3),1)
destP = np.full((N,3),1)
for i in range(0,N):
    srcP[i][0] = kp2[matches[i].trainIdx].pt[0]
    srcP[i][1] = kp2[matches[i].trainIdx].pt[1]
    destP[i][0] = kp1[matches[i].queryIdx].pt[0]
    destP[i][1] = kp1[matches[i].queryIdx].pt[1]
h = compute_homography(srcP,destP)
height,weight = desk.shape[:2]
res_h = cv.warpPerspective(cover,h,(weight,height))
for i in range(0,res_h.shape[0]):
    for j in range(0,res_h.shape[1]):
        if res_h[i][j] == 0:
            res_h[i][j] = desk[i][j]
h_ = h
cv.imshow("homography_cv.png",res_h)
#cv.waitKey(0)

#rd.seed(15)
rd.seed(19)
ir = 0
rsh = np.full((3,3),0)
srcP = np.full((100, 3), 1)
destP = np.full((100, 3), 1)
for i in range(0, 100):
    srcP[i][0] = kp2[matches[i].trainIdx].pt[0]
    srcP[i][1] = kp2[matches[i].trainIdx].pt[1]
    destP[i][0] = kp1[matches[i].queryIdx].pt[0]
    destP[i][1] = kp1[matches[i].queryIdx].pt[1]
st = time.process_time()
rsh, ir = compute_homography_ransac(srcP,destP,1)
ed = time.process_time()
id = np.argmax(ir,axis=1)
h = rsh[id[0]]

height,weight = desk.shape[:2]
res_r = cv.warpPerspective(cover,h,(weight,height))
res_r2 = cv.warpPerspective(hp,h,(weight,height))
for i in range(0,res_r.shape[0]):
    for j in range(0,res_r.shape[1]):
        if res_r[i][j] == 0:
            res_r[i][j] = desk[i][j]

cv.imshow("ransac_cv.png",res_r)
##################################################hp.jpg

for i in range(0,res_r2.shape[0]):
    for j in range(0,res_r2.shape[1]):
        if res_r2[i][j] == 0:
            res_r2[i][j] = desk[i][j]

cv.imshow("ransac_hp.png",res_r2)

###########################################stiching

desk = cv.imread('diamondhead-10.png', cv.IMREAD_GRAYSCALE)
cover = cv.imread('diamondhead-11.png', cv.IMREAD_GRAYSCALE)
cv.waitKey(0)
out = None
orb = cv.ORB_create()
kp1, des1 = orb.detectAndCompute(desk, None)
kp2, des2 = orb.detectAndCompute(cover, None)

tmp = 0
tIdx = 0
qIdx = 0
matches = []
for i in range(0,len(kp1)):
    for j in range(0,len(kp2)):
        tmp = cv.norm(des1[i],des2[j],cv.NORM_HAMMING)
        if j == 0:
            dist = tmp
        elif dist > tmp:
            dist = tmp
            tIdx = i
            qIdx = j
    match = cv.DMatch(tIdx,qIdx,dist)
    matches.append(match)

listtodel = []
for i in range(0,len(matches)):
    x = matches[i].trainIdx
    y = matches[i].queryIdx
    d = matches[i].distance

    for j in range(0,len(matches)):
        if i == j:
            continue
        if matches[i].trainIdx == matches[j].trainIdx and matches[i].queryIdx != matches[j].queryIdx:
            if matches[j].distance > d:
                listtodel.append(j)
        if matches[i].queryIdx == matches[j].queryIdx and matches[i].trainIdx != matches[j].trainIdx:
            if matches[j].distance > d:
                listtodel.append(j)

listtodel = list(set(listtodel))
listtodel.sort(reverse=True)
for i in listtodel:
    del matches[i]

matches = sorted(matches, key=lambda x: x.distance)
out = cv.drawMatches(desk,kp1,cover,kp2,matches[:10],out,flags=2) #feature mathching
N = 15
srcP = np.full((N,3),1)
destP = np.full((N,3),1)
for i in range(0,N):
    srcP[i][0] = kp2[matches[i].trainIdx].pt[0]
    srcP[i][1] = kp2[matches[i].trainIdx].pt[1]
    destP[i][0] = kp1[matches[i].queryIdx].pt[0]
    destP[i][1] = kp1[matches[i].queryIdx].pt[1]
h = compute_homography(srcP,destP)
height,weight = desk.shape[:2]
res_h = cv.warpPerspective(cover,h,(weight,height))
for i in range(0,res_h.shape[0]):
    for j in range(0,res_h.shape[1]):
        if res_h[i][j] == 0:
            res_h[i][j] = desk[i][j]
h_ = h
rd.seed(5)
ir = 0
rsh = np.full((3,3),0)
srcP = np.full((100, 3), 1)
destP = np.full((100, 3), 1)
for i in range(0, 100):
    srcP[i][0] = kp2[matches[i].trainIdx].pt[0]
    srcP[i][1] = kp2[matches[i].trainIdx].pt[1]
    destP[i][0] = kp1[matches[i].queryIdx].pt[0]
    destP[i][1] = kp1[matches[i].queryIdx].pt[1]
st = time.process_time()
rsh, ir = compute_homography_ransac(srcP,destP,1)
ed = time.process_time()
id = np.argmax(ir,axis=1)
h = rsh[id[0]]
height,weight = desk.shape[:2]
res_r = cv.warpPerspective(cover,h_,(int(1.3*weight),height))
idx = 0

for i in range(0,cover.shape[0]):
    for j in range(0,cover.shape[1]):
        if res_r[i][j] == 0:
            if res_r[i][j+1] != 0:
                res_r[i][j] = (1/2)*res_r[i][j+1] + (1/2)*desk[i][j]
            res_r[i][j] = desk[i][j]
        if i>10 and j > 326 and j < 335 :
            res_r[i][j] = desk[i-5][j]


cv.imshow("diamondhead_ransac.png",res_r)
cv.waitKey(0)