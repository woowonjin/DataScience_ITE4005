import pandas as pd
import sys
import math
from collections import deque
import numpy as np
import random


def is_in_distance(x1, y1, x2, y2, eps):
    dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    return dist <= eps


input_file = sys.argv[1]
n = int(sys.argv[2])
eps = float(sys.argv[3])
min_pts = int(sys.argv[4])

df = pd.read_csv(input_file, sep="\t", names=["index", "x", "y"])
print(df.shape)
data = df.values.tolist()
data_dict = {}
for d in data:
    data_dict[int(d[0])] = (d[1], d[2])
clusters = []
visited_idx = []
cnt = 0

while(data_dict.keys()):
    # random_idx = random.randint(0, len(data_dict)-1)
    criteria = list(data_dict.keys())[0]
    queue = deque([criteria])  # core candidate
    clusters.append([])
    while(queue):
        criteria_candidate = queue.popleft()
        if not criteria_candidate in visited_idx:
            clusters[cnt].append(criteria_candidate)
            visited_idx.append(criteria_candidate)
            x1, y1 = data_dict[criteria_candidate]
            temp_list = []  # for check min_pts
            for key in data_dict.keys():
                x2, y2 = data_dict[key]
                if is_in_distance(x1, y1, x2, y2, eps):
                    temp_list.append(key)
            if len(temp_list) >= min_pts:
                for key in temp_list:
                    queue.append(key)
    for d in clusters[cnt]:
        del data_dict[d]
    cnt += 1
clusters.sort(key=lambda x: len(x), reverse=True)
for i in range(n):
    temp = np.array(clusters[i])
    temp_df = pd.DataFrame(data=temp)
    file = input_file.split(".")
    temp_df.to_csv(file[0]+"_cluster_"+str(i)+".txt", header=None, index=None)
    print(temp.shape)
