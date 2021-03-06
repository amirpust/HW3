import pandas as pd
from math import log2 as log
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


class Node:

    def __init__(self, examples, parent):
        self.total = len(examples)
        self.num_of_M = len([x for x in examples if x[1] == "M"])
        self.num_of_B = self.total - self.num_of_M
        self.right = None
        self.left = None
        self.examples = examples
        self.selected_feature = None
        self.sep = -1  # go left if smaller then this bound
        self.entropy_val = self.entropy()
        self.diagnosis = 'M' if self.num_of_M >= 0.1 * self.num_of_B else 'B'
        self.parent = parent
        self.is_leaf = False

    def entropy(self):
        p = self.num_of_M / self.total
        if p == 0 or p == 1:
            return 0
        return -p * log(p) - (1-p) * log(1-p)

    def set_children(self, left, right):
        self.left = left
        self.right = right


class ID3:

    def __init__(self):
        self.root: Node

    def fit(self, examples, f: np.ndarray):
        self.root = Node(examples, None)
        self.TDIDT(self.root, f)

    def TDIDT(self, current_node: Node, f: np.ndarray):
        if current_node.total == 0 or current_node.num_of_B == 0 or current_node.num_of_M == 0:
            current_node.is_leaf = True
            return None

        selected_feat = self.maxIG(f, current_node)
        current_examples = current_node.examples
        left_son_info = [tup for tup in current_examples if f[selected_feat[1]][tup[0]] < selected_feat[0]]
        right_son_info = [tup for tup in current_examples if f[selected_feat[1]][tup[0]] >= selected_feat[0]]
        left = Node(left_son_info, current_node)
        right = Node(right_son_info, current_node)
        current_node.sep = selected_feat[0]
        current_node.selected_feature = selected_feat[1]
        current_node.set_children(left, right)

        self.TDIDT(left, f)
        self.TDIDT(right, f)

    @staticmethod
    def entropy(p):
        if p == 0 or p == 1:
            return 0
        return -p * log(p) - (1-p) * log(1-p)

    # will return the col of the selected feature
    def maxIG(self,features, current_node: Node):
        ig = []
        separators = []

        for feat in features:
            current_ig, separator = self.IG_feature(current_node, feat)
            ig.append(current_ig)
            separators.append(separator)

        best = np.argmax(np.flip(np.array(ig)))
        best = len(ig) - best - 1
        return [separators[best], best]

    ''' needs to receive a feature and then returns the best bound with the best ig '''
    def IG_feature(self, current_node: Node, feature: np.ndarray):

        sorted_col = np.array(list(dict.fromkeys([feature[int(loc[0])] for loc in current_node.examples])))
        sorted_col.sort()
        separators = [(sorted_col[i]+sorted_col[i+1])/2 for i in range(0, len(sorted_col) - 1)]
        ig = []

        for sep in separators:
            under_sep = [tup for tup in current_node.examples if feature[tup[0]] < sep]
            under_M = len([x for x in under_sep if x[1] == 'M'])
            under_B = len(under_sep) - under_M
            over_M = current_node.num_of_M - under_M
            over_B = current_node.num_of_B - under_B
            p_under = under_M / (under_B + under_M) if under_B + under_M > 0 else 0
            p_over = over_M / (over_B + over_M) if over_B + over_M > 0 else 0
            ig.append(current_node.entropy_val - (1 / current_node.total) * ((under_B + under_M) * self.entropy(p_under)
                                                                             + (over_B + over_M) * self.entropy(p_over)))
        best = np.argmax(ig)
        return ig[int(best)], separators[int(best)]

    def predict(self, example: np.ndarray):
        iterator = self.root
        while not iterator.is_leaf:
            iterator = iterator.left if example[iterator.selected_feature] < iterator.sep else iterator.right
        return iterator.diagnosis

    def print_tree(self, node: Node):
        if node is None:
            return
        self.print_tree(node.left)
        print("the feat: " + str(node.selected_feature) + " the bound: " + str(node.sep))
        self.print_tree(node.right)


class MID3(ID3):
    def __init__(self, M):
        super().__init__()
        self.M_limit = M

    def TDIDT(self, current_node: Node, f: np.ndarray):
        if current_node.total < self.M_limit:
            current_node.diagnosis = current_node.parent.diagnosis
            current_node.is_leaf = True
            return None
        elif current_node.num_of_M == 0 or current_node.num_of_B == 0:
            current_node.is_leaf = True
            return None
        selected_feat = self.maxIG(f, current_node)
        current_examples = current_node.examples
        left_son_info = [tup for tup in current_examples if f[selected_feat[1]][tup[0]] < selected_feat[0]]
        right_son_info = [tup for tup in current_examples if f[selected_feat[1]][tup[0]] >= selected_feat[0]]
        left = Node(left_son_info, current_node)
        right = Node(right_son_info, current_node)
        current_node.sep = selected_feat[0]
        current_node.selected_feature = selected_feat[1]
        current_node.set_children(left, right)

        self.TDIDT(left, f)
        self.TDIDT(right, f)

def run_test(algo, to_print):
    test_set = pd.read_csv('test.csv')
    correct_cnt = 0
    for index, row in test_set.iterrows():
        if row[0] == algo.predict(np.array(row)[1:]):
            correct_cnt += 1
    if to_print:
        print(correct_cnt / len(test_set.index))


'''this function expects that all the data will be available to it i.e the training and the 
test data should be in the right format as shown below'''
def experiment():
    M_list = [10, 12, 15, 18, 20, 25]
    kf = KFold(n_splits=5, shuffle=True, random_state=316397843)

    kf.get_n_splits(diag_train)
    success_rate = []
    for m in M_list:
        avg = 0
        for sub_train, sub_test in kf.split(data_diag):
            to_train_on = [data_diag[train_idx] for train_idx in sub_train]
            to_test_on = [training_set_data.iloc[test_idx] for test_idx in sub_test]
            false_positive = 0
            false_negative = 0
            mid3 = MID3(m)
            mid3.fit(to_train_on, features)
            for row in to_test_on:
                algorithm_diag = mid3.predict(np.array(row)[1:])
                if row[0] == 'B' and algorithm_diag == 'M':
                    false_positive += 1
                elif row[0] == 'M' and algorithm_diag == 'B':
                    false_negative += 1
            avg += (0.1 * false_positive + false_negative) / len(to_test_on)
        success_rate.append(avg / 5)
# TODO : delete this part
    plt.plot(M_list, success_rate)
    plt.xlabel("M values")
    plt.ylabel("success rate")
    plt.show()
    return M_list[int(np.argmin(np.array(success_rate)))]


training_set_data = pd.read_csv("train.csv")
diag_train = np.array(training_set_data['diagnosis'])
data_diag = [(index, diag_train[index]) for index in range(len(diag_train))]
features = np.array([training_set_data[col] for col in training_set_data if col != 'diagnosis'])

# sec 4.3 ------------
best = experiment()
M_id3 = MID3(best)
M_id3.fit(data_diag, features)
test_set = pd.read_csv('test.csv')
false_positive = 0
false_negative = 0
for index, row in test_set.iterrows():
    algorithm_diag = M_id3.predict(np.array(row)[1:])
    if row[0] == 'B' and algorithm_diag == 'M':
        false_positive += 1
    elif row[0] == 'M' and algorithm_diag == 'B':
        false_negative += 1

loss = (0.1 * false_positive + false_negative) / len(test_set.index)
# TODO : delete
print("fp = " + str(false_positive) + " fn = " + str(false_negative))
run_test(M_id3, True)
print(loss)
