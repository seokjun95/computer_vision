import numpy as np
import random
import struct


s = './sift/sift'
input_sift = np.ndarray((0,128))
file_len = np.ndarray((1000,))
file_idx = np.ndarray((1000,))
random.seed(1)

for i in range(100000,101000):
    f = open(s+str(i),'rb')
    f_int = np.fromfile(f,dtype=np.uint8).reshape(-1,128)
    file_len[i-100000] = f_int.shape[0]
    if i-100000 == 0:
        file_idx[i-100000] = 0
    else:
        file_idx[i-100000] = file_idx[i-100000-1]+file_len[i-100000-1]
    input_sift = np.vstack((input_sift, f_int))


start = random.sample(range(0,input_sift.shape[0]),8)
centroid = np.ndarray((8,128))
for i in range(8):
    centroid[i] = input_sift[start[i]]
sampling = random.sample(range(0, input_sift.shape[0]), 100000)
sampling = np.asarray(sampling)

while True:
    clustering_set = [[], [], [], [], [], [], [], []]
    nearest_idx = 0
    nearest = 0
    for i in range(len(sampling)):
        for j in range(8):
            dist = np.linalg.norm(input_sift[sampling[i]] - centroid[j])
            if j == 0:
                nearest = dist
                nearest_idx = 0
            elif dist < nearest:
                nearest = dist
                nearest_idx = j
        clustering_set[nearest_idx].append(sampling[i])
    diff = 0
    for i in range(0,8):
        sum = 0
        for j in range(len(clustering_set[i])):
            sum += input_sift[clustering_set[i][j]]
        mean = sum/(len(clustering_set[i]))
        diff += mean - centroid[i]
        centroid[i] = mean

    if np.linalg.norm(diff) <= 15:
        break



adapt_cent = np.ndarray((input_sift.shape[0],1))
final_desc = np.ndarray((0,1024))

nearest = 0
nearest_idx = 0
for i in range(1000): # for each file
    tmp = np.zeros((1, 1024))

    for j in range(int(file_idx[i]), int(file_idx[i]+file_len[i])): # for each descriptor
        for k in range(8): # find centeroid
            dist = np.linalg.norm(input_sift[j]-centroid[k])
            if k ==0:
                nearest = dist
            elif dist < nearest:
                nearest = dist
                nearest_idx = k
        adapt_cent[j][0] = nearest_idx
    for j in range(int(file_idx[i]), int(file_idx[i]+file_len[i])): # for each descriptor

        if adapt_cent[j][0] == 0:
            tmp[0,:128] += (input_sift[j]-centroid[0])
        elif adapt_cent[j][0] == 1:
            tmp[0, 128:256] += (input_sift[j] - centroid[1])
        elif adapt_cent[j][0] == 2:
            tmp[0, 256:384] += (input_sift[j] - centroid[2])
        elif adapt_cent[j][0] == 3:
            tmp[0, 384:512] += (input_sift[j] - centroid[3])
        elif adapt_cent[j][0] == 4:
            tmp[0, 512:640] += (input_sift[j] - centroid[4])
        elif adapt_cent[j][0] == 5:
            tmp[0, 640:768] += (input_sift[j] - centroid[5])
        elif adapt_cent[j][0] == 6:
            tmp[0, 768:896] += (input_sift[j] - centroid[6])
        elif adapt_cent[j][0] == 7:
            tmp[0, 896:1024] += (input_sift[j] - centroid[7])
    final_desc = np.append(final_desc,tmp,axis=0)

out = open('A3_2015313846.des','wb')
var = struct.pack("ii",1000,1024)
out.write(var)

final_desc = np.array(final_desc,dtype=np.float32).tobytes()
out.write(final_desc)