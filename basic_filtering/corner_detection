from Function import *
import cv2
import numpy as np
import time

lenna = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
shapes = cv2.imread('shapes.png', cv2.IMREAD_GRAYSCALE)

if not os.path.exists('./result'):
    os.mkdir('./result')

start_time = time.process_time()
lenna_r = compute_coner_response(lenna)
end_time = time.process_time()
print("elapsed time for lenna compute coner response : ",end_time-start_time)
cv2.imshow("part3_corner_raw_lenna.png",lenna_r)
cv2.imwrite('./result/part3_corener_raw_lenna.png',lenna_r*255)


start_time = time.process_time()
shapes_r = compute_coner_response(shapes)
end_time = time.process_time()
print("elapsed time for shapes compute coner response : ",end_time-start_time)
cv2.imshow("part3_corner_raw_shapes.png",shapes_r)
cv2.imwrite('./result/part3_corener_raw_shapes.png',shapes_r*255)
cv2.waitKey(0)

cv2.imwrite('lenna_gray.png',lenna)
cv2.imwrite('shapes_gray.png',shapes)
lenna_dot = cv2.imread("lenna_gray.png",cv2.IMREAD_COLOR)
shapes_dot = cv2.imread("shapes_gray.png",cv2.IMREAD_COLOR)
for i in range(2, lenna.shape[0]-2):
    for j in range(2, lenna.shape[1]-2):
        if lenna_r[i][j] > 0.1:
            cv2.circle(lenna_dot, (j,i), 1, (0,255,0), 1)

for i in range(2, shapes.shape[0] - 2):
    for j in range(2, shapes.shape[1] - 2):
        if shapes_r[i][j] > 0.1:
            cv2.circle(shapes_dot, (j, i), 1, (0, 255, 0), 1)

cv2.imshow("lenna dot",lenna_dot)
cv2.imwrite("./result/part3_corner_bin_lenna.png",lenna_dot)
cv2.imshow("shapes dot",shapes_dot)
cv2.imwrite("./result/part3_corner_bin_shapes.png",shapes_dot)
cv2.waitKey(0)

lenna_dot = cv2.imread("lenna_gray.png",cv2.IMREAD_COLOR)
shapes_dot = cv2.imread("shapes_gray.png",cv2.IMREAD_COLOR)
start_time = time.process_time()
lenna_suppressed = non_maximum_suppression_win(lenna_r,11)
end_time = time.process_time()
print("elapsed time for lenna non maximum suppression win : ",end_time-start_time)
start_time = time.process_time()
shapes_suppressed = non_maximum_suppression_win(shapes_r,11)
end_time = time.process_time()
print("elapsed time for shapes non maximum suppression win : ",end_time-start_time)
for i in range(5, lenna.shape[0]-5):
    for j in range(5, lenna.shape[1]-5):
        if lenna_suppressed[i][j]>0.1:
            cv2.circle(lenna_dot,(j,i),5,(0,255,0), 1)
for i in range(5, shapes.shape[0]-5):
    for j in range(5, shapes.shape[1]-5):
        if shapes_suppressed[i][j]>0.1:
            cv2.circle(shapes_dot,(j,i),5,(0,255,0), 1)
cv2.imshow("lenna_suppressed",lenna_dot)
cv2.imwrite("./result/part3_corner_sup_lenna.png",lenna_dot)
cv2.imshow("shapes suppressed",shapes_dot)
cv2.imwrite("./result/part3_corner_sup_shapes.png",shapes_dot)
cv2.waitKey(0)
