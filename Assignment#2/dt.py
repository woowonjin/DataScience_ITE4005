import sys
import pandas as pd
import numpy as np
import copy


def calculate_info(p, n):  # n : 총갯수,  p : 원소 갯수
    if p == 0 or n == 0:
        return 0
    arg1 = p/n
    return round(-arg1*np.log2(arg1), 10)


class LeafNode:
    def __init__(self, result, name):
        self.result = result
        self.name = name
        # print(result, "LeafNode 추가")


class Node:
    # Class Variable : data, attr_arr, main_attr, data_split_criteria, node_arr
    def split_info(self, attr):
        all_cnt = self.data.shape[0]
        temp_data = self.data.groupby(attr).size().values
        split = 0
        for value in temp_data:
            split += -value/all_cnt*np.log2(value/all_cnt)
        return split

    def __init__(self, data, attr_arr, name):
        self.data = data  # 각 노드마다 가지는 training data
        self.attr_arr = attr_arr  # 각 노드마다 가지는 남은 attribute array
        self.name = name
        if not self.data[self.attr_arr[-1]].shape[0]:
            print("What the Fuck!!")
            return
        all_gain = self.get_all_data_gain()
        max_gain = -1
        max_gain_attr = None
        for i in range(len(self.attr_arr)-1):
            attr_gain = round(
                (all_gain - self.calculate_gain(self.attr_arr[i]))/self.split_info(self.attr_arr[i]), 10)
            if(attr_gain >= max_gain):
                max_gain = attr_gain
                max_gain_attr = self.attr_arr[i]
        self.main_attr = max_gain_attr
        self.data_split_criteria = self.data[self.main_attr].unique()
        # print("일반 Node 추가")
        # print("Node name : ", self.name)
        # print("Main Attr : ", self.main_attr)
        # print(self.data)
        next_node_attr_arr = copy.deepcopy(self.attr_arr)
        next_node_attr_arr = np.delete(next_node_attr_arr, np.where(
            next_node_attr_arr == self.main_attr))
        self.node_arr = []
        for criteria in self.data_split_criteria:
            new_data = self.data.groupby(self.main_attr).get_group(criteria)
            next_node_data = new_data.drop(columns=[self.main_attr])
            # 모든 라벨이 같을때 LeafNode
            if(len(next_node_data[next_node_attr_arr[-1]].unique()) == 1):
                result = new_data[next_node_attr_arr[-1]].values[0]
                temp_node = LeafNode(result, criteria)
                self.node_arr.append(temp_node)
            elif(len(next_node_attr_arr) == 1):  # 더이상 attribute가 없을때 LeafNode
                result_dict = {}
                for res in next_node_data[next_node_attr_arr[-1]].values:
                    if res in result_dict:
                        result_dict[res] += 1
                    else:
                        result_dict[res] = 1
                result = None
                max_cnt = 0
                for key in result_dict.keys():
                    if result_dict[key] >= max_cnt:
                        max_cnt = result_dict[key]
                        result = key
                temp_node = LeafNode(result, criteria)
                self.node_arr.append(temp_node)
            elif(not new_data.shape):  # 데이터가 하나도 없을때 -> LeafNode를 어떻게 할지 고민
                temp_data = self.data[self.attr_arr[-1]].size()
                print(temp_data)
            else:  # 일반 node추가
                temp_node = Node(next_node_data, copy.deepcopy(
                    next_node_attr_arr), criteria)
                self.node_arr.append(temp_node)

    def get_all_data_gain(self):
        all_df_about_label = self.data.groupby(self.attr_arr[-1])
        all_df_about_label_cnt = all_df_about_label.size().values
        all_cnt = self.data.shape[0]

        if len(all_df_about_label_cnt) == 1:
            return 0
        else:
            info_all = 0
            for val in all_df_about_label_cnt:
                info_all += calculate_info(val, all_cnt)
            return info_all

    def calculate_gain(self, attr):
        attr_df = self.data[[attr, self.attr_arr[-1]]]
        attr_values = attr_df[attr].unique()
        attr_all_cnt = attr_df.groupby(attr).size().values.sum()
        attr_gain = 0
        for value in attr_values:  # 31...40, <=30, >40
            df_about_value = attr_df[attr_df[attr] == value]
            df_about_value_count = df_about_value.shape[0]
            df_about_value = df_about_value.groupby(
                self.attr_arr[-1]).size().values
            if len(df_about_value) == 1:
                attr_gain += 0
            else:
                for val in df_about_value:
                    attr_gain += round(df_about_value_count/attr_all_cnt *
                                       calculate_info(val, df_about_value_count), 10)
        return attr_gain


train_file = sys.argv[1]
test_file = sys.argv[2]
result_file = sys.argv[3]

train_df = pd.read_csv(train_file, sep="\t", dtype=str)
test_df = pd.read_csv(test_file, sep="\t", dtype=str)
# ans_df = pd.read_csv(result_file, sep="\t", dtype=str)  # for test

all_attr = train_df[train_df.columns.values[-1]].unique()
majority_all = train_df.groupby(train_df.columns.values[-1])
max_count = 0
majority_voting_result = ""
for a in all_attr:
    val = majority_all.size()[a]
    if val >= max_count:
        max_count = val
        majority_voting_result = a


def traverse_node(root, data_dict):
    node = root
    while True:
        if node.__class__ == LeafNode:
            return node.result
        is_in = False
        for temp_node in node.node_arr:
            if temp_node.name == data_dict[node.main_attr]:
                node = temp_node
                is_in = True
                break
        if not is_in:  # 트레이닝 데이터셋으로 만들었을때 해당 노드가 없는경우 -> 부모노드에서 다수결로 한다 -> 실습에 물어보고 결정
            data = node.data
            attr_arr_temp = data[data.columns.values[-1]].unique()
            data = data.groupby(node.attr_arr[-1])
            max_count_temp = 0
            majority_voting_result_temp = ""
            major_arr = []
            for a in attr_arr_temp:
                val = data.size()[a]
                if val > max_count_temp:
                    max_count_temp = val
                    majority_voting_result_temp = a
            # return "e"  # for error data
            return majority_voting_result_temp


# def traverse_tree(root):
#     node = root
#     if(node.__class__ == Node):
#         print("Node :", node.name, ", main_attr :", node.main_attr)
#     elif(node.__class__ == LeafNode):
#         print("LeafNode :", node.name)

#     if node.__class__ == LeafNode:
#         return
#     for n in node.node_arr:
#         traverse_tree(n)


error_data = []


def find_result(root, test_data, result_attr):
    attr_arr = test_data.columns.values
    test_data[result_attr] = ""
    for i in range(test_data.shape[0]):
        data = test_data.iloc[i]
        data_dict = {}
        for attr in attr_arr:
            data_dict[attr] = data[attr]
        result = traverse_node(root, data_dict)
        if result == "e":
            error_data.append(data)
        test_data.at[i, result_attr] = result
    test_data.to_csv(result_file, sep="\t", index=False)
    # 테스팅 코드
    # my_ans = test_data[result_attr]
    # real_ans = ans_df[result_attr]
    # all_cnt = 0
    # correct_cnt = 0
    # for i in range(test_data.shape[0]):
    #     all_cnt += 1
    #     if(my_ans.iloc[i] == real_ans.iloc[i]):
    #         correct_cnt += 1
    # print(f"{correct_cnt}/{all_cnt} -> {round(correct_cnt/all_cnt*100, 3)}% accuracy")


# Error 데이터 제외하고 정확도 측정
# def find_result(root, test_data, result_attr):
#     attr_arr = test_data.columns.values
#     test_data[result_attr] = ""
#     for i in range(test_data.shape[0]):
#         data = test_data.iloc[i]
#         data_dict = {}
#         for attr in attr_arr:
#             data_dict[attr] = data[attr]
#         result = traverse_node(root, data_dict)
#         if result == "e":
#             error_data.append(data)
#         test_data.at[i, result_attr] = result
#         # print(result)

#     my_ans = test_data[result_attr]
#     real_ans = ans_df[result_attr]
#     all_cnt = 0
#     correct_cnt = 0
#     for i in range(test_data.shape[0]):
#         if my_ans.iloc[i] != "e":
#             all_cnt += 1
#             if(my_ans.iloc[i] == real_ans.iloc[i]):
#                 correct_cnt += 1
#     print(f"{correct_cnt}/{all_cnt} -> {round(correct_cnt/all_cnt*100, 3)}% accuracy")


if __name__ == "__main__":
    root = Node(train_df, train_df.columns.values, None)
    # print(root.__class__ == Node)
    # traverse_tree(root)
    find_result(root, test_df, train_df.columns.values[-1])
    # print(error_data)
    # print(error_data)
    # for d in error_data:
    #     d_arr = d.values
    #     tf = train_df[(train_df["buying"] == d_arr[0]) & (train_df["maint"] == d_arr[1]) & (train_df["doors"] == d_arr[2]) & (
    #         train_df["persons"] == d_arr[3]) & (train_df["lug_boot"] == d_arr[4]) & (train_df["safety"] == d_arr[5])]
    #     print(tf)
