import numpy as np


filename = "8_raw.txt"
x = np.loadtxt(filename, dtype = float, unpack = False)
print("original shape of input:", x.shape)

x = np.array(x)

x = x.reshape(1,x.shape[0])

# y = np.zeros([14557-12, 13])
y = np.zeros([4806 - 12, 13])
# y = np.zeros([14557 - 6, 13])

print("shape after change:", x.shape)

j = 0
for i in range(6, 4806 - 6):
    y[j,:] = x[0, i - 6 : i + 7]
    j = j + 1

    if j > y.shape[0] - 1:
        break

# print y[14552-6 - 1,:]
print(y[4806-12-1, 7])
np.savetxt("8_edited.txt", y)

# f = open("1_edited.txt",'w')
# f.write(y)
# f.close()