from _timing import timeit
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import os
import time


start = time.time()

data_path = r"acc_datas"
log_path =r"logs"
data_paths = [os.path.join(data_path, name) for name in os.listdir(data_path)]
all_data = []

file_name = (str(data_paths[0]))


for file in data_paths:
    data = []
    with open(file, "r") as f:
        all_data = np.asarray(
            [(lambda line: list(map(float, line.split("\t"))))(line)
             for line in f.readlines()],
            dtype=float)

all_data[:, 1] *= 9.81


# plt.plot(all_data[:, 0], all_data[:, 1], '-')
# plt.show()
#
# exit()

f = interpolate.interp1d(all_data[:, 0], all_data[:, 1])


def S_acc (T: float, dt0: float):

    # timeit["A1"]

    global max_acc

    v0 = 0.0
    x0 = 0.0
    w = 2.0 * np.pi / T
    w2 = w ** 2

    v_all = [v0]
    x_all = [x0]

    i = -1
    t0 = all_data[0,0]
    t_all = [t0]

    # timeit["A1"]


    dt = dt0 * 10 * T
    t_max = all_data[-1][0]

    t = np.arange(t0, t_max, dt)

    zdd = f(t)

    zdd_ort_array =  0.5 * (np.roll(zdd, -1) + zdd)

    for i, t0 in enumerate(t[:-1]):
        t1 = t[i+1]

        # timeit["A3"]

        # zdd_ort = 0.5 * (zdd0 + zdd1)
        zdd_ort = zdd_ort_array[i]
        v1 = v0 - dt * (zdd_ort + w2 * x0 + 0.1 * w * v0)
        v_ort = 0.5 * (v0 + v1)
        x1 = x0 + dt * v_ort

        # timeit["A3"]


        # timeit["A4"]

        v_all.append(v1)
        x_all.append(x1)
        t_all.append(t1)

        # timeit["A4"]

        v0 = v1
        x0 = x1

    # plt.plot(t_all, x_all, '-')
    # plt.show()

    max_acc = abs(max(x_all, key=abs))

    return max_acc

minT = 1
maxT = 15
spec_interval = 0.1
dt = 0.0001

spec_acc = []
spec_t = (np.arange(minT,maxT,spec_interval))


i=1
for t in np.arange(minT,maxT,spec_interval):
    perc = (i/len(spec_t))*100
    i += 1
    print("Process: %", perc)
    S_acc(t, dt)
    spec_acc.append(max_acc)



d = interpolate.interp1d(spec_t, spec_acc)
spec_t_array = np.arange(minT, maxT-1, 0.0001)
spec_array = d(spec_t_array)


plt.plot(spec_t_array, spec_array, '-')
timestr = time.strftime("%Y%m%d_%H%M%S")
plt.savefig(log_path + "\\" +   timestr + "_Spectral_Acc")

plt.clf()

plt.plot(all_data[:, 0], all_data[:, 1], '-')
plt.savefig(log_path + "\\" +   timestr + "_Ground_Acc")

log = [
       "Date: ", timestr, '\n',
       "Data File: ", file_name.split('\\')[-1], '\n',
       "dt: ", str(dt), '\n',
       "Min T: ", str(minT), '\n',
       "Max T: ", str(maxT), '\n',
       "Spectral Interval: ", str(spec_interval)
       ]


complete_name = os.path.join(log_path, timestr + "_aLog" + ".txt")
with open(complete_name, 'w') as f:
    for line in log:
        f.write(line)

end = time.time()
print("Toplam sure:", int((end - start)//60), "dk", (end-start)%60, "sn" )

# Test feautre deneme

# timeit.report()

# plt.show()