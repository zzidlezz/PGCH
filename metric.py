import torch
from torch.autograd import Variable
import numpy as np
from scipy.linalg import hadamard
import random

def compress(train_loader, test_loader, model_gcn, train_dataset, test_dataset):
    re_BI = list([])
    re_BT = list([])
    for _, (data_I, data_T,_, _) in enumerate(train_loader):
        var_data_I = Variable(data_I.cuda())
        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        img_common, txt_common, code_I, code_T, img_real, img_fake, txt_real, txt_fake= model_gcn(var_data_I,var_data_T)



        code_I = torch.sign(code_I)
        re_BI.extend(code_I.cpu().data.numpy())


        code_T = torch.sign(code_T)
        re_BT.extend(code_T.cpu().data.numpy())

    qu_BI = list([])
    qu_BT = list([])
    for _, (data_I, data_T, _, _) in enumerate(test_loader):
        var_data_I = Variable(data_I.cuda())
        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        img_common, txt_common, code_I, code_T, img_real, img_fake, txt_real, txt_fake= model_gcn(var_data_I,var_data_T)

        code_I = torch.sign(code_I)
        qu_BI.extend(code_I.cpu().data.numpy())




        code_T = torch.sign(code_T)
        qu_BT.extend(code_T.cpu().data.numpy())

    re_BI = np.array(re_BI)
    re_BT = np.array(re_BT)
    re_L = train_dataset.train_labels

    qu_BI = np.array(qu_BI)
    qu_BT = np.array(qu_BT)
    qu_L = test_dataset.train_labels
    return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L



def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    leng = B2.shape[1]
    distH = 0.5 * (leng - np.dot(B1, B2.transpose()))
    return distH

def calculate_top_map(qu_B, re_B, qu_L, re_L, topk):
    """
    :param qu_B: {-1,+1}^{mxq} query bits
    :param re_B: {-1,+1}^{nxq} retrieval bits
    :param qu_L: {0,1}^{mxl} query label
    :param re_L: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    num_query = qu_L.shape[0]
    topkmap = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap


def zero2eps(x):
    x[x == 0] = 1
    return x

def normalize(affnty):
    col_sum = zero2eps(np.sum(affnty, axis=1)[:, np.newaxis])
    row_sum = zero2eps(np.sum(affnty, axis=0))
    out_affnty = affnty/col_sum
    in_affnty = np.transpose(affnty/row_sum)
    return in_affnty, out_affnty

def affinity_tag_multi(tag1: np.ndarray, tag2: np.ndarray):
    aff = np.matmul(tag1, tag2.T)
    affinity_matrix = np.float32(aff)
    in_aff, out_aff = normalize(affinity_matrix)
    return in_aff, out_aff

def get_hash_targets(n_class, bit):
        H_K = hadamard(bit)
        diffMat = np.abs(np.diff(H_K)) / 2
        invTimes = np.sum(diffMat, axis=1)
        W_K = H_K[invTimes.argsort(), :]
        W_2K = np.concatenate((W_K, -W_K), 0)
        hash_targets = torch.from_numpy(W_2K[:n_class]).float()
        if W_2K.shape[0] < n_class:
            hash_targets.resize_(n_class, bit)
            for k in range(20):
                for index in range(W_2K.shape[0], n_class):
                    ones = torch.ones(bit)
                    sa = random.sample(list(range(bit)), bit // 2)
                    ones[sa] = -1
                    hash_targets[index] = ones
                c = []
                for i in range(n_class):
                    for j in range(n_class):
                        if i < j:
                            TF = sum(hash_targets[i] != hash_targets[j])
                            c.append(TF)
                c = np.array(c)
                if c.min() > bit / 4 and c.mean() >= bit / 2:
                    print(c.min(), c.mean())
                    break
        return hash_targets