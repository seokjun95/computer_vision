from Function import *
import cv2
import numpy as np

lenna = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
shapes = cv2.imread('shapes.png', cv2.IMREAD_GRAYSCALE)

if not os.path.exists('./result'):
    os.mkdir('./result')

kernel = get_gaussian_filter_2d(7, 1.5)
lenna_img = cross_correlation_2d(lenna, kernel)
start_time = time.process_time()
lenna_mag, lenna_dir = compute_image_gradient(lenna_img)
end_time = time.process_time()
print("Elapsed time for compute lenna gradient: ",end_time-start_time)
cv2.imshow("part2_edge_raw_lenna.png",lenna_mag/255)
cv2.imwrite("./result/part2_edge_raw_lenna.png",lenna_mag)


kernel = get_gaussian_filter_2d(7, 1.5)
lenna_img = cross_correlation_2d(shapes, kernel)
start_time = time.process_time()
shapes_mag, shapes_dir = compute_image_gradient(lenna_img)
end_time = time.process_time()
print("Elapsed time for compute shapes gradient: ",end_time-start_time)
cv2.imshow("part2_edge_raw_shapes.png",shapes_mag/255)
cv2.imwrite("./result/part2_edge_raw_shpes.png",shapes_mag)
cv2.waitKey(0)
