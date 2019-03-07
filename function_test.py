import pf_speed
import pandas as pd
import numpy as np
path = r"D:\download\gps_example.csv"
def add_random_error(lat, n, pos_error, neg_error):
    for i in range(n):
        pos = np.random.randint(len(lat))
        if np.random.rand() < 0.5:
            lat[pos] = lat[pos] + pos_error
        elif lat[pos] > neg_error:
            lat[pos] = lat[pos] - neg_error
        else:
            lat[pos] = 0
    return(lat)
gps_example = pd.read_csv(path, sep = ",")
speed = gps_example.iloc[:,3].copy()
speed = add_random_error(speed, 10, np.random.rand() * 60, np.random.rand() * 60)
import matplotlib.pyplot as plt
t = [i for i in range(len(speed))]
x_est_out = pf_speed.particle_filter_speed(speed)
plt.figure(figsize=(16,24), facecolor = "white")
graph_num = int(np.floor(len(speed) / 500))
for i in range(graph_num):
    plt.subplot(graph_num,1,1+i)
    start=i*500
    plt.plot(t[start:start+500], speed[start:start+500], linestyle="-", linewidth=1, label='origin')
    plt.plot(t[start:start+500], x_est_out[start:start+500], linestyle="-.", linewidth=1, label='filter')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
plt.show()