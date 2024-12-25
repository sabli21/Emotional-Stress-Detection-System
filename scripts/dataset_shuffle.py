

import sklearn
import numpy as np






x = np.loadtxt("total_input.txt", dtype = float, unpack = False)



y = np.loadtxt("total_label_v2.txt",dtype = float, unpack = False)




y = y.astype(int)

x_shuffle, y_shuffle = sklearn.utils.shuffle(x, y, random_state = 0)

# np.savetxt("total_input_shuffle.txt", x_shuffle, fmt = '%10.8f')
# np.savetxt("total_label_v2_shuffle.txt", y_shuffle,fmt = '%10.4f')