# -*- coding: utf-8 -*-
import sys
import itertools
import copy

minimum_support_num = int(sys.argv[1])
input_file = sys.argv[2]
output_file = sys.argv[3]

is_stop = False

# 모든 거래내역이 담길 변수
DB_transactions = []

# min_support를 만족하는 모든 items
frequent_items = []

# item의 길이가 1인것 부터 시작, 현재 얼마 길이의 아이템셋을 보고있는지를 알려주는 지표이다.
current_item_len = 1

cnt = 0
# DB에 들어있는 transaction들 정보를 DB_transactions array에 다 담기
with open(input_file, "r") as f:
    while True:
        line = f.readline()
        if not line:
            break
        cnt += 1
        without_line_change = line.split("\n")
        transaction = list(map(int, without_line_change[0].split("\t")))
        DB_transactions.append(transaction)
    f.close()

minimum_support = float(minimum_support_num/100 * cnt)


def remove_item_about_support(item_dict):
    result = []
    for item in item_dict:
        if item_dict[item] >= minimum_support:
            result.append(item)
    if len(result) == 0:
        is_stop = True
    else:
        return result


def list_to_set(item_list):
    result = []
    for item in item_list:
        result.append(set(item))
    return result


def join(length):
    if length == 2:
        result = list(itertools.combinations(
            frequent_items[current_item_len-2], current_item_len))
        result_list = list(map(list, result))
        result_set = list_to_set(result_list)
        if(len(result_set) == 0):
            is_stop = True
        return result_set
    else:  # 원소가 3 개이상
        elements = []
        k_items = frequent_items[current_item_len-2]
        for k_item in k_items:
            k_item_list = list(k_item)
            for item in k_item_list:
                if item not in elements:
                    elements.append(item)
        combinations_result = list(itertools.combinations(elements, length))
        remove_result = copy.deepcopy(combinations_result)
        for l in combinations_result:  # 원소 3개 짜리 items라고 생각
            l_combinations = list(itertools.combinations(l, length-1))
            for i in l_combinations:  # 원소 2개 짜리 items라고 생각
                i_set = set(i)
                if i_set not in frequent_items[length-2]:
                    remove_result.remove(l)
                    break
        remove_result_set = list_to_set(remove_result)
        if len(remove_result_set) == 0:
            is_stop = True
        return remove_result_set


def execute_apriori():
    global current_item_len
    global DB_transactions
    while True:
        dict_for_item_and_count = {}
        if current_item_len == 1:
            for transac in DB_transactions:
                for item in transac:
                    if item not in dict_for_item_and_count:
                        dict_for_item_and_count[item] = 1
                    else:
                        dict_for_item_and_count[item] += 1
            result = remove_item_about_support(dict_for_item_and_count)  # list
            frequent_items.append(result)
            current_item_len += 1
        else:  # 원소 갯수가 2개 이상
            items = join(current_item_len)  # sets
            if is_stop:
                break
            dict_for_item_and_count = {}
            for item in items:
                item_cnt = 0
                for transac in DB_transactions:
                    transac_combinations = list(
                        itertools.combinations(transac, current_item_len))
                    transac_sets = list_to_set(transac_combinations)
                    if item in transac_sets:
                        item_cnt += 1
                dict_for_item_and_count[tuple(item)] = item_cnt
            # print(dict_for_item_and_count)
            result = remove_item_about_support(dict_for_item_and_count)
            if result == None:
                break
            result_set = list_to_set(result)
            frequent_items.append(result_set)
            current_item_len += 1


def association_rule():
    max_frequent_len = len(frequent_items)
    DB_set = list_to_set(DB_transactions)
    DB_len = len(DB_transactions)
    file_output = open(output_file, "w")
    for i in range(1, max_frequent_len):  # 빈번하게 발생하는 모든 elements
        for element in frequent_items[i]:
            element_len = i+1
            for_iter = 1
            support_cnt = 0
            for transac in DB_set:
                if transac >= element:
                    support_cnt += 1

            while element_len > for_iter:
                element_combinations = list(
                    itertools.combinations(element, element_len-for_iter))
                element_comb_set = list_to_set(element_combinations)
                for comb in element_comb_set:
                    left = element - comb  # left side, comb is right side
                    left_cnt = 0
                    right_cnt = 0
                    for transac in DB_set:
                        if transac >= left:
                            left_cnt += 1
                            if transac >= comb:
                                right_cnt += 1
                    support = format(support_cnt/DB_len * 100, ".2f")
                    confidence = format(right_cnt/left_cnt*100, ".2f")
                    output = str(left) + "\t" + str(comb) + "\t" + \
                        str(support) + "\t" + str(confidence) + "\n"
                    file_output.write(output)
                for_iter += 1


execute_apriori()
association_rule()
