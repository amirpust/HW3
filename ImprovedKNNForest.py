import pandas as pd
from math import log2 as log
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import random


class Node:

    def __init__(self, examples, parent):
        self.total = len(examples)
        self.num_of_M = len([x for x in examples if x[1] == "M"])
        self.num_of_B = self.total - self.num_of_M
        self.right = None
        self.left = None
        self.examples = examples
        self.selected_feature = None
        self.sep = -1                                                 # go left if smaller then this bound
        self.entropy_val = self.entropy()
        self.diagnosis = 'M' if self.num_of_M >= self.num_of_B else 'B'
        self.parent = parent
        self.is_leaf = False

    # def is_leaf(self):
    #     return self.num_of_M == 0 or self.num_of_B == 0

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
        if len(ig) == 0:
            return -1, sorted_col[-1]

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


class KNN:

    def __init__(self, N, K, P, weigh=False, M=0):
        self.M = M
        self.p = P
        self.num_of_d_trees = N
        self.K = K
        self.trees = []
        self.centroid = []
        self.weigh = weigh




    def fit(self, examples, feat):
        for i in range(self.num_of_d_trees):
            random.seed(i)
            current_examples = random.sample(examples, int(len(examples) * self.p) + 1)
            self.calc_centroid(current_examples, feat)
            decision_tree = MID3(self.M)
            decision_tree.fit(current_examples, feat)
            self.trees.append(decision_tree)

    def calc_centroid(self, examples, feat: np.ndarray):
        num_of_examples = len(examples)
        current_centroid = []
        for f in feat:
            avg = 0
            for index, diag in examples:
                avg += f[index]
            avg /= num_of_examples
            current_centroid.append(avg)
        self.centroid.append(np.array(current_centroid))

    def predict(self, to_predict_on):
        euclid_distances = []
        for tree_index in range(len(self.centroid)):
            euclid_dist = np.linalg.norm(self.centroid[tree_index] - to_predict_on)
            euclid_distances.append((tree_index, euclid_dist))

        euclid_distances.sort(key=lambda x: x[1])

        x_min = euclid_distances[0][1]
        x_max = euclid_distances[self.K - 1][1]

        num_of_M = 0
        num_of_B = 0
        # removed the multiplication of i
        for i in range(self.K):
            classification = self.trees[euclid_distances[i][0]].predict(to_predict_on)
            if classification == 'M':
                num_of_M += self.normalize(x_min, x_max, euclid_distances[i][1]) if self.weigh else 1
            else:
                num_of_B += self.normalize(x_min, x_max, euclid_distances[i][1]) if self.weigh else 1
        return 'M' if num_of_M > num_of_B else 'B'

    def normalize(self, x_min, x_max, val):
        return 1 - ((val - x_min) / (x_max - x_min))

def run_test(algo, to_print, test_group):

    correct_cnt = 0
    for row in test_group:
        if row[0] == algo.predict(np.array(row)[1:]):
            correct_cnt += 1
    if to_print:
        print(correct_cnt / len(test_group))
    return correct_cnt / len(test_group)


def choose_params(N, train_data, feats, probability, K, M):
    kf = KFold(n_splits=5, random_state=316397843, shuffle=True)
    kf.get_n_splits(train_data)
    best_parameters = []
    for num_of_trees in N:
        for p in probability:
            for m in M:
                for k in K:
                    avg = 0
                    for sub_train, sub_test in kf.split(train_data):
                        to_train_on = [data_diag[train_idx] for train_idx in sub_train]
                        to_test_on = [training_set_data.iloc[test_idx] for test_idx in sub_test]
                        knn = KNN(N=num_of_trees, K=int(k * num_of_trees), P=p, weigh=True,M=m)
                        knn.fit(to_train_on, feats)
                        accuracy = run_test(knn, False, to_test_on)
                        avg += accuracy
                    avg /= 5
                    print("accuracy = " + str(avg) + " k = " + str(k) + " N = " + str(num_of_trees) + " p = " + str(p) +
                          " M = " + str(m))
                    best_parameters.append((avg, k, num_of_trees, p, m))
    return max(best_parameters, key=lambda x: x[0])


# sec 7 -----------

training_set_data = pd.read_csv("train.csv")
diag_train = np.array(training_set_data['diagnosis'])
n = len(diag_train)
N = [15, 20, 25, 30]
P = [0.3, 0.4, 0.5, 0.6, 0.7]
K = [0.25, 0.35, 0.45, 0.6, 0.7]
M = [1, 5, 10, 15]
data_diag = [(index, diag_train[index]) for index in range(len(diag_train))]
features = np.array([training_set_data[col] for col in training_set_data if col != 'diagnosis'])
best = choose_params(N, data_diag, features, P, K, M)
print("k = " + str(best[1]) + " N = " + str(best[2]) + " P = " + str(best[3]) + " M = " + str(best[-1]))
N = 15
K = 0.7
P = 0.7
knn = KNN(N=N, K=int(np.ceil(N * K)), P=P, weigh=True)
knn.fit(examples=data_diag, feat=features)
test_set = pd.read_csv('test.csv')
run_test(knn, True, [row for idx, row in test_set.iterrows()])


# take 2 : lets try to use the minmax over the centroid.