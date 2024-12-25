import numpy as np
filename = "8_1.txt"

label = np.zeros((4806 - 12, 1),dtype = int)

tag = np.loadtxt(filename, dtype = int, unpack = False)
print tag.shape


index_1 = np.where(tag[0 : tag.shape[0] - 6, 1] == 1)
index_1 = np.array(index_1,dtype=int)
index_1 = index_1.reshape(index_1.shape[1],1)

print index_1

print index_1.shape

# print tag[384,0]


for i in range(len(index_1[:,0])):
    label[tag[index_1[i, 0], 0] - 6 - 1, 0] = 1
#label = label.T

np.savetxt("8_1label.txt", label,fmt = '%10.4f')



