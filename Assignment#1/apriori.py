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

# DB transaction의 갯수를 구하기 위해
cnt = 0

# DB에 들어있는 transaction들 정보를 DB_transactions array에 다 담기
with open(input_file, "r") as f:
    while True:
        line = f.readline()
        if not line:
            break
        cnt += 1
        without_line_change = line.split("\n")
        # DB의 값들이 모두 integer 형태이기 떄문에 integer형태로 바꿔 주었다.
        transaction = list(map(int, without_line_change[0].split("\t")))
        DB_transactions.append(transaction)
    f.close()

# minimum_support값을 퍼센트로 받았기 때문에 계산의 편리성을 위해 갯수로 바꿨다
minimum_support = float(minimum_support_num/100 * cnt)


"""
각 elements값들이 DB에 들어있는 갯수를 {원소: 갯수} 형태로 가지고 있는 Dictionary를 인자로 받아 minimum_support
를 넘지 못하는 elements들은 거르고, 만족하는 원소들의 list를 반환하는 함수이다
"""
def remove_item_about_support(item_dict):
    result = []
    for item in item_dict:
        if item_dict[item] >= minimum_support:
            result.append(item)
    if len(result) == 0:
        is_stop = True
    else:
        return result


"""
예를들어 [1, 2]가 [1, 2, 3]에 들어있는지와 같은 확인은 순서에 상관없어야하고, 편리성을 위해 set으로 바꾸는게 필요해서
list를 set으로 바꾸는 함수를 만들었다
"""
def list_to_set(item_list):
    result = []
    for item in item_list:
        result.append(set(item))
    return result


"""
조합에 해당하는 함수로, 현재 가진 원소들에서 나올수 있는 모든 조합을 만들고 그 조합들 중에서, 각각 마다 만약
subset이 해당 갯수의 frequent set에 하나라도 포함되지않으면 지우는 함수이다.
"""
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


"""
원소가 1개부터 늘어나는 형식이므로 frequent한 원소들은 추가하고, 또 원소의 갯수도 증가시켜준다. 메인함수라고 볼수았다.
frequent_items list에는 0번 index에는 길이 1짜리 frequent item set list, 1번 index에는 길이 2짜리 frequent item set list, 이런식으로 저장된다.
"""
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


"""
association rule을 만드는 함수로, 만약 {1, 2, 3}이라는 원소가 있다면, 이와 관련된 길이 1, 2짜리 모든 subset을 만들었고,
그것과의 차집합을 구해 차집합 -> subset 식의 모든 association rule을 만들어 support와 confidence 를 구했다.
"""
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
# print(frequent_items)
association_rule()
