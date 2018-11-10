import random
def get_random_group_order(group_count=20):
    groups_order = list(range(1, group_count+1))
    random.shuffle(groups_order)
    return groups_order


random.seed(10)
group_order = get_random_group_order(20)
print(group_order)
