from Function import *
import cv2
import numpy as np

lenna = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
shapes = cv2.imread('shapes.png',cv2.IMREAD_GRAYSCALE)

if not os.path.exists('./result'):
    os.mkdir('./result')
#1.print gaussian kernel
print(get_gaussian_filter_1d(5,1))
print(get_gaussian_filter_2d(5,1))

#compare seq 1d ,2d
start_time = time.process_time()
seq1d = cross_correlation_1d(cross_correlation_1d(lenna,get_gaussian_filter_1d(5,1).T),get_gaussian_filter_1d(5,1))
end_time = time.process_time()
print("Elapsed time for sequential 1D:",end_time-start_time)
start_time = time.process_time()
seq2d = cross_correlation_2d(lenna, get_gaussian_filter_2d(5,1))
end_time = time.process_time()
print("Elapsed time for 2D:",end_time - start_time)
diff = abs(seq1d - seq2d)
print("Difference between seq1D and 2D : ",np.sum(diff))
cv2.imshow("diff",diff)
cv2.waitKey(0)

# 9 filtered lenna
p00 = cv2.resize(cross_correlation_2d(lenna,get_gaussian_filter_2d(5,1)), dsize = (0, 0), fx = 1/3, fy = 1/3)
cv2.putText(p00, '5x5 s=1', (10, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 0, 1)
p01 = cv2.resize(cross_correlation_2d(lenna,get_gaussian_filter_2d(5,6)), dsize = (0, 0), fx = 1/3, fy = 1/3)
cv2.putText(p01, '5x5 s=6', (10, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 0, 1)
p02 = cv2.resize(cross_correlation_2d(lenna,get_gaussian_filter_2d(5,11)), dsize = (0, 0), fx = 1/3, fy = 1/3)
cv2.putText(p02, '5x5 s=11', (10, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 0, 1)

p10 = cv2.resize(cross_correlation_2d(lenna,get_gaussian_filter_2d(11,1)), dsize = (0, 0), fx = 1/3, fy = 1/3)
cv2.putText(p10, '11x11 s=1', (10, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 0, 1)
p11 = cv2.resize(cross_correlation_2d(lenna,get_gaussian_filter_2d(11,6)), dsize = (0, 0), fx = 1/3, fy = 1/3)
cv2.putText(p11, '11x11 s=6', (10, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 0, 1)
p12 = cv2.resize(cross_correlation_2d(lenna,get_gaussian_filter_2d(11,11)), dsize = (0, 0), fx = 1/3, fy = 1/3)
cv2.putText(p12, '11x11 s=11', (10, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 0, 1)

p20 = cv2.resize(cross_correlation_2d(lenna,get_gaussian_filter_2d(17,1)), dsize = (0, 0), fx = 1/3, fy = 1/3)
cv2.putText(p20, '17x17 s=1', (10, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 0, 1)
p21 = cv2.resize(cross_correlation_2d(lenna,get_gaussian_filter_2d(17,6)), dsize = (0, 0), fx = 1/3, fy = 1/3)
cv2.putText(p21, '17x17 s=6', (10, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 0, 1)
p22 = cv2.resize(cross_correlation_2d(lenna,get_gaussian_filter_2d(17,11)), dsize = (0, 0), fx = 1/3, fy = 1/3)
cv2.putText(p22, '17x17 s=11', (10, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 0, 1)

r1 = np.hstack((p00, p01, p02))
r2 = np.hstack((p10, p11, p12))
r3 = np.hstack((p20, p21, p22))
arr = np.vstack((r1,r2,r3))
cv2.imshow('lenna.png',arr/255)
cv2.imwrite('./result/result_gaussian_filtered_lenna.png',arr)

# 9 filtered shape
p00 = cv2.resize(cross_correlation_2d(shapes,get_gaussian_filter_2d(5,1)), dsize = (0, 0), fx = 1/3, fy = 1/3)
cv2.putText(p00, '5x5 s=1', (10, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 0, 1)
p01 = cv2.resize(cross_correlation_2d(shapes,get_gaussian_filter_2d(5,6)), dsize = (0, 0), fx = 1/3, fy = 1/3)
cv2.putText(p01, '5x5 s=6', (10, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 0, 1)
p02 = cv2.resize(cross_correlation_2d(shapes,get_gaussian_filter_2d(5,11)), dsize = (0, 0), fx = 1/3, fy = 1/3)
cv2.putText(p02, '5x5 s=11', (10, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 0, 1)

p10 = cv2.resize(cross_correlation_2d(shapes,get_gaussian_filter_2d(11,1)), dsize = (0, 0), fx = 1/3, fy = 1/3)
cv2.putText(p10, '11x11 s=1', (10, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 0, 1)
p11 = cv2.resize(cross_correlation_2d(shapes,get_gaussian_filter_2d(11,6)), dsize = (0, 0), fx = 1/3, fy = 1/3)
cv2.putText(p11, '11x11 s=6', (10, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 0, 1)
p12 = cv2.resize(cross_correlation_2d(shapes,get_gaussian_filter_2d(11,11)), dsize = (0, 0), fx = 1/3, fy = 1/3)
cv2.putText(p12, '11x11 s=11', (10, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 0, 1)

p20 = cv2.resize(cross_correlation_2d(shapes,get_gaussian_filter_2d(17,1)), dsize = (0, 0), fx = 1/3, fy = 1/3)
cv2.putText(p20, '17x17 s=1', (10, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 0, 1)
p21 = cv2.resize(cross_correlation_2d(shapes,get_gaussian_filter_2d(17,6)), dsize = (0, 0), fx = 1/3, fy = 1/3)
cv2.putText(p21, '17x17 s=6', (10, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 0, 1)
p22 = cv2.resize(cross_correlation_2d(shapes,get_gaussian_filter_2d(17,11)), dsize = (0, 0), fx = 1/3, fy = 1/3)
cv2.putText(p22, '17x17 s=11', (10, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 0, 1)

r1 = np.hstack((p00, p01, p02))
r2 = np.hstack((p10, p11, p12))
r3 = np.hstack((p20, p21, p22))
arr = np.vstack((r1,r2,r3))
cv2.imshow('shapes.png',arr/255)
cv2.imwrite('./result/result_gaussian_filtered_shapes.png',arr)
cv2.waitKey(0)


