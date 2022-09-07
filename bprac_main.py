import random
import numpy as np
import pandas as pd
import scipy
import math
from scipy import integrate
from scipy.special import gamma
import evaluation

project_data_path = r'.\mov100k.csv'
train_data_path = r'.\mov100k_train.csv'
test_data_path = r'.\mov100k_test.csv'


def load_data(path):
    project_data = pd.read_csv(path, header=0, dtype='str', sep=',')

    user_index_dict = {}
    userlist = list(project_data['user'].unique())
    u_index = 0
    for user in userlist:
        user_index_dict[user] = u_index
        u_index += 1

    item_index_dict = {}
    itemlist = list(project_data['item'].unique())
    i_index = 0
    for item in itemlist:
        item_index_dict[item] = i_index
        i_index += 1

    user_count = len(userlist)
    item_count = len(itemlist)

    count = project_data.shape[0]
    print('user_count:%d' % user_count, 'item_count:%d' % item_count)
    return user_count,userlist,user_index_dict,item_count,itemlist,item_index_dict
user_count,userlist,user_index_dict,item_count,itemlist,item_index_dict = load_data(project_data_path)


def load_train(path):
    '''
    :param path: train data_path
    :return: train dict
    '''
    datadict = {}
    datafram = pd.read_csv(path, header=0, sep=',', dtype='str')
    for i in datafram.itertuples():
        user = getattr(i, 'user')
        item = getattr(i, 'item')
        rating = getattr(i, 'rating')
        datadict.setdefault(user, {})
        datadict[user][item] = rating
    return datadict
train_data = load_train(train_data_path)
print('train ok')

def load_test(path):
    '''
    :param path: test data path
    :return: np_array shape = user_count * item_count
    '''
    testdata_fram = pd.read_csv(path, header=0, sep=',', dtype='str')
    test_data = np.zeros((user_count, item_count))
    for i in testdata_fram.itertuples():
        userno = user_index_dict[getattr(i, 'user')]
        itemno = item_index_dict[getattr(i, 'item')]
        test_data[userno, itemno] = 1
    return test_data
lable = load_test(test_data_path)
print('test ok')



def prior(data):
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    fre = np.zeros(item_count)
    for u in data.keys():
        for i in data[u].keys():
            item_index = item_index_dict[i]
            fre[item_index] +=1
    mean = np.mean(fre)
    std = np.std(fre)
    prior = [sigmoid((i - mean) / std) for i in fre]
    return prior
prior = prior(train_data)

random.seed(0)
np.random.seed(0)


class BPRAC:
    lr = 0.05
    reg = 0.6
    reg1 = 0.01
    d = 27
    u1, u2 = 0, 0
    sigma1, sigma2 = reg ** 0.5, reg ** 0.5

    U = np.random.normal(loc=u1,
                         scale=sigma1,
                         size=[user_count, d])
    V = np.random.normal(loc=u2,
                         scale=sigma2,
                         size=[item_count, d])
    confidence = np.zeros((user_count, item_count))

    T = 10
    train_count = 1000

    def pdf(self, x, d, reg):
        x = abs(x) / reg + 1e-10
        a = reg * math.pi ** 0.5 * gamma(d / 2)
        b = (x / 2) ** ((d - 1) / 2)
        c = scipy.special.kn((d - 1) / 2, x)
        return b * c / a

    def cdf(self, x, d, reg):
        y, err = integrate.quad(self.pdf, -10, x, args=(d, reg))
        return y

    def g(self, x, n, k):
        c = math.factorial(n) / (math.factorial(k - 1) * math.factorial(n - k))
        return c * self.cdf(x, self.d, self.reg) ** (k - 1) * self.pdf(x, self.d, self.reg) * (1 - self.cdf(x, self.d, self.reg)) ** (n - k)


    def sigmoid(self, x):
        if x >= 0:
            return 1 / (1 + np.exp(-x))
        else:
            return np.exp(x) / (1 + np.exp(x))



    def train(self, train_dict):
        counter1 = 0
        # T-round interations
        for t in range(self.T):
            print('---------BPRAC: %d-round interations------------' % t)
            # Expectation: updating c_{ui}
            counter = 0
            score_raw = np.mat(self.U) * np.mat(self.V.T)
            for user in train_dict:
                interaction = train_dict[user]
                user_no = user_index_dict[user]
                for item in interaction:
                    rating = int(train_dict[user][item])
                    k = 2
                    n = 2
                    item_no = item_index_dict[item]
                    xui = score_raw[user_no, item_no]
                    cui = self.g(xui, n, k) * rating
                    self.confidence[user_no, item_no] = cui
                    counter += 1
                    if counter % 2000 == 0:
                        print('Expectation: updating %d-confidence' % counter)

            # Maximization: updating theta
            for k in range(self.train_count):
                counter1 += 1
                if counter1 % 100 == 0:
                    print('Maximization: BPRAC %d-training' % counter1)
                for user in range(user_count):
                    user = random.sample(userlist,1)[0]
                    keys = list(train_dict[user].keys())
                    i_item = random.sample(keys,1)[0]

                    j_item = random.sample(itemlist,1)[0]
                    while j_item in keys:
                        j_item = random.sample(itemlist,1)[0]

                    u = user_index_dict[user]
                    i = item_index_dict[i_item]

                    j = item_index_dict[j_item]

                    r_ui = np.dot(self.U[u], self.V[i].T)
                    r_uj = np.dot(self.U[u], self.V[j].T)
                    r_uij = r_ui - r_uj
                    cui = self.confidence[u, i]

                    loss_func = 1 - self.sigmoid(r_uij)
                    # update U and V
                    self.U[u] += self.lr * cui * (loss_func * (self.V[i] - self.V[j]) -  self.reg1*self.U[u])
                    self.V[i] += self.lr * cui * (loss_func * self.U[u] -  self.reg1*self.V[i])
                    self.V[j] += self.lr * cui * (loss_func * (-self.U[u]) -  self.reg1*self.V[j])
        return (np.mat(self.U) * np.mat(self.V.T))


raw_score = BPRAC().train(train_data)
score = evaluation.erase(raw_score,train_data,user_index_dict,item_index_dict)
#Top-k evaluation'
print(evaluation.topk(score,lable,5), evaluation.mapk(score,lable,5))
print(evaluation.topk(score,lable,10),evaluation.mapk(score,lable,10))
