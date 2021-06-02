#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
train_file = sys.argv[1]
test_file = sys.argv[2]
# train_file = "u4.base"
# test_file = "u4.test"

k = 50  # number of iteration
alpha = 0.25  # learning rate
_lambda = 0.1  # parameter for regularizer
patience = 5  # number of patience that not improved from previous version
attr_num = 12  # number of columns in factorized matrix
column_names = ["user_id", "item_id", "rating", "time_stamp"]

dataframe = pd.read_csv(train_file, sep="\t", names=column_names, header=None)
testframe = pd.read_csv(test_file, sep="\t", names=column_names, header=None)

dataframe = dataframe.drop(["time_stamp"], axis=1)
testframe = testframe.drop(["time_stamp"], axis=1)

last_movie = dataframe["item_id"].max()


# In[2]:


train_data, val_data = train_test_split(
    dataframe, test_size=0.05, random_state=123)
print(train_data)
print(val_data)


# In[3]:


df_table = train_data.pivot("user_id", "item_id", "rating")
for i in range(1, last_movie+1):
    if not i in df_table.columns:
        df_table[i] = np.NaN
df_table = df_table[[i for i in range(1, last_movie+1)]]
print(df_table)


# In[4]:


pre_use_mat = df_table.replace([1.0, 2.0, 3.0, 4.0, 5.0], 1.0).values
pre_use_val = copy.deepcopy(val_data.values)
for temp in pre_use_val:
    temp[2] = 1
print(pre_use_val)
row = pre_use_mat.shape[0]
col = pre_use_mat.shape[1]
rated_indexes = np.argwhere(pre_use_mat == 1)
print(rated_indexes.shape)


# In[5]:


def mean_squared_error_with_bias(y, p, q, b_p, b_q, mean):
    return ((y-mean-b_p-b_q-np.dot(p, q))**2+_lambda*np.sqrt(np.sum(np.square(p))) + _lambda*np.sqrt(np.sum(np.square(q))) + _lambda*(b_p**2) + _lambda*(b_q**2) + 0.e-10)


def mf_uninteresting(index_arr, validation_set):
    p = np.random.rand(row, attr_num)  # number of parameter : row * 3
    q = np.random.rand(col, attr_num)  # number of parameter : col * 3
    b_p = np.random.rand(row, 1)
    b_q = np.random.rand(col, 1)
    min_rmse_error = 9999999
    best_p = None
    best_q = None
    best_b_p = None
    best_b_q = None
    not_improved_cnt = 0
    for _ in range(k):
        d_p = [[np.zeros(attr_num), 0] for _ in range(row)]
        d_q = [[np.zeros(attr_num), 0] for _ in range(col)]
        d_b_p = [[0, 0] for _ in range(row)]
        d_b_q = [[0, 0] for _ in range(col)]
        error = 0
        for i, j in index_arr:
            p_i = p[i]
            q_j = q[j]
            b_p_i = b_p[i][0]
            b_q_j = b_q[j][0]
            a = np.dot(p_i, q_j)
            # values of derivation p
            d_p[i][0] += _lambda*p_i - (1-b_p_i-b_q_j-1-a)*q_j
            d_p[i][1] += 1  # cnt
            d_q[j][0] += _lambda*q_j - (1-b_p_i-b_q_j-1-a)*p_i
            d_q[j][1] += 1
            d_b_p[i][0] += _lambda*b_p_i - (1-b_p_i-b_q_j-1-a)
            d_b_p[i][1] += 1
            d_b_q[j][0] += _lambda*b_q_j - (1-b_p_i-b_q_j-1-a)
            d_b_q[j][1] += 1
            error += mean_squared_error_with_bias(
                1, p[i], q[j], b_p_i, b_q_j, 1)

        # current value - derivation_average«
        for i in range(row):
            if d_p[i][1] != 0:
                p[i] -= alpha*(d_p[i][0]/d_p[i][1])
                b_p[i][0] -= alpha*(d_b_p[i][0]/d_b_p[i][1])
        # current value - derivation_average
        for j in range(col):
            if d_q[j][1] != 0:
                q[j] -= alpha*(d_q[j][0]/d_q[j][1])
                b_q[j][0] -= alpha*(d_b_q[j][0]/d_b_q[j][1])
        error_ = error/len(index_arr)
        print("Training_error : ", error_)
        result_temp = np.dot(p, q.transpose()) + 1
        result_temp += b_p
        result_temp += b_q.transpose()
        rmse_error = 0
        for user, movie, rating in validation_set:
            rmse_error += (result_temp[user-1][movie-1]-rating) ** 2
        rmse_error = np.sqrt(rmse_error/len(validation_set))
        print("Validation Error : ", rmse_error)
        if rmse_error < min_rmse_error:
            min_rmse_error = rmse_error
            best_p = p
            best_q = q
            best_b_p = b_p
            best_b_q = b_q
        else:
            not_improved_cnt += 1
            print(f"Did not improved from {min_rmse_error} to {rmse_error}")
            if not_improved_cnt >= patience:
                print("Early Stopped!!")
                return best_p, best_q, best_b_p, best_b_q
    return best_p, best_q, best_b_p, best_b_q


# In[6]:


uninteresting_p, uninteresting_q, uninteresting_b_p, uninteresting_b_q = mf_uninteresting(
    rated_indexes, pre_use_val)


# In[7]:


pre_use_mat_result = np.dot(uninteresting_p, uninteresting_q.transpose()) + 1
pre_use_mat_result += uninteresting_b_p
pre_use_mat_result += uninteresting_b_q.transpose()
print(pre_use_mat_result)
uninteresting_rating_index = np.argwhere(pre_use_mat_result < 0.8)
print(uninteresting_rating_index.shape)


# In[8]:


original_rating_mat = df_table.fillna(-1).values
for index in uninteresting_rating_index:
    original_rating_mat[index[0]][index[1]] = 0
result_rated_indexes = np.argwhere(original_rating_mat >= 0)
rating_sum = 0
cnt = 0
for index in result_rated_indexes:
    rating_sum += original_rating_mat[index[0]][index[1]]
    cnt += 1
rating_mean = rating_sum / cnt


# In[9]:


def mf_training(index_arr, data_arr, validation_set, mean):
    p = np.random.rand(row, attr_num)  # number of parameter : row * attr_num
    q = np.random.rand(col, attr_num)  # number of parameter : col * attr_num
    b_p = np.random.rand(row, 1)
    b_q = np.random.rand(col, 1)
    min_rmse_error = 9999999
    best_p = None
    best_q = None
    best_b_p = None
    best_b_q = None
    not_improved_cnt = 0
    for _ in range(k):
        d_p = [[np.zeros(attr_num), 0] for _ in range(row)]
        d_q = [[np.zeros(attr_num), 0] for _ in range(col)]
        d_b_p = [[0, 0] for _ in range(row)]
        d_b_q = [[0, 0] for _ in range(col)]
        acc_error = 0
        rmse_error = 0
        for i, j in index_arr:
            p_i = p[i]
            q_j = q[j]
            b_p_i = b_p[i][0]
            b_q_j = b_q[j][0]
            a = np.dot(p_i, q_j)
            # values of derivation p
            d_p[i][0] += _lambda*p_i - (data_arr[i][j]-b_p_i-b_q_j-mean-a)*q_j
            d_p[i][1] += 1  # cnt
            d_q[j][0] += _lambda*q_j - (data_arr[i][j]-b_p_i-b_q_j-mean-a)*p_i
            d_q[j][1] += 1
            d_b_p[i][0] += _lambda*b_p_i - (data_arr[i][j]-b_p_i-b_q_j-mean-a)
            d_b_p[i][1] += 1
            d_b_q[j][0] += _lambda*b_q_j - (data_arr[i][j]-b_p_i-b_q_j-mean-a)
            d_b_q[j][1] += 1
            acc_error += mean_squared_error_with_bias(
                data_arr[i][j], p[i], q[j], b_p_i, b_q_j, mean)

        # current value - derivation_average«
        for i in range(row):
            if d_p[i][1] != 0:
                p[i] -= alpha*(d_p[i][0]/d_p[i][1])
                b_p[i][0] -= alpha*(d_b_p[i][0]/d_b_p[i][1])
        # current value - derivation_average
        for j in range(col):
            if d_q[j][1] != 0:
                q[j] -= alpha*(d_q[j][0]/d_q[j][1])
                b_q[j][0] -= alpha*(d_b_q[j][0]/d_b_q[j][1])

        acc_error_ = acc_error/len(index_arr)
        print("Training Error : ", acc_error_)
        # for validation
        result_temp = np.dot(p, q.transpose())+mean
        result_temp += b_p
        result_temp += b_q.transpose()
        for user, movie, rating in validation_set:
            rmse_error += (result_temp[user-1][movie-1]-rating) ** 2
        rmse_error = np.sqrt(rmse_error/len(validation_set))
        print("Validation Error : ", rmse_error)
        if rmse_error < min_rmse_error:
            min_rmse_error = rmse_error
            best_p = p
            best_q = q
            best_b_p = b_p
            best_b_q = b_q
        else:
            not_improved_cnt += 1
            print(f"Did not improved from {min_rmse_error} to {rmse_error}")
            if not_improved_cnt >= patience:
                print("Early Stopped!!")
                return best_p, best_q, best_b_p, best_b_q
    return best_p, best_q, best_b_p, best_b_q


# In[10]:


valid_set = val_data.values
print(valid_set)
p, q, b_p, b_q = mf_training(
    result_rated_indexes, original_rating_mat, valid_set, rating_mean)


# In[11]:


recommend_result = np.dot(p, q.transpose()) + rating_mean
recommend_result += b_p
recommend_result += b_q.transpose()
recommend_result = np.where(recommend_result < 1.0, 1, recommend_result)
recommend_result = np.where(recommend_result > 5.0, 5, recommend_result)
print(recommend_result)


# In[12]:


rmse = 0
cnt = 0
for_write = []
for user, movie, rating in testframe.values:
    if user > row or movie > col:
        for_write.append([user, movie, 3])
        rmse += (rating-3)**2
    else:
        for_write.append([user, movie, recommend_result[user-1][movie-1]])
        rmse += (rating-recommend_result[user-1][movie-1])**2
rmse = np.sqrt(rmse/testframe.shape[0])
print(rmse)
df_write = pd.DataFrame(for_write)
df_write.to_csv(train_file+"_prediction.txt",
                sep="\t", index=None, header=None)


# In[ ]:
