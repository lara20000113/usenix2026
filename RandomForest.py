# -*- coding:utf-8 -*-
# from sklearn.ensemble import RandomForestClassifier
# import numpy as np
# from sklearn.externals.six import StringIO
import queue
# from sklearn.metrics.ranking import roc_auc_score
# import os
import random
from decimal import *
# import time
import math
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

def getType(ch):
    if ch == 1:
        return 0
    if ch >= ord('0') and ch <= ord('9'):
        return 1
    elif ch >= ord('a') and ch <= ord('z') or ch >= ord('A') and ch <= ord('Z'):
        return 2
    else:
        return 3
def RF(train_file, test_file, sample_file, guess_file):
    def str2vec(str):
        vec = st_vec[:]
        for c in str:
            vec.append(ord(c))
        vec.append(0)
        return vec
    def getFeaturevec(vec, last_pos):
        #ch = vec[-1]
        l = vec.__len__() - gram
        cur_len = l + gram - 1 - last_pos
        rtn = [l, cur_len]
        for c in vec[-gram:]:
            if c in feature_dic.keys():
                rtn += feature_dic[c]
            else:
                rtn +=feature_dic[ord('?')]
        return rtn
    def randPick(rf, vec, last_pos):
        rand_pro = random.random()
        p = 0
        sum = 0
        feature_vec = getFeaturevec(vec, last_pos)
        cur_type = getType(vec[-1])

        for prob in rf.predict_proba([feature_vec])[0]:
            sum += (prob + smooth) / smooth_classNum
            if sum > rand_pro:
                g_type = getType(rfclass[p])
                g_pos = last_pos
                if cur_type != g_type:
                    g_pos = vec.__len__() - 1
                return rfclass[p], (prob + smooth) / smooth_classNum, g_pos
            p += 1
        return
    def calPro(rf, vec, last_pos):
        pro = 1
        for i in range(gram, vec.__len__()):
            feature_vec = getFeaturevec(vec[:i], last_pos)
            p = 0
            for prob in rf.predict_proba([feature_vec])[0]:
                if rfclass[p] == vec[i]:
                    pro *= (prob + smooth) / smooth_classNum
                    cur_type = getType(vec[i - 1])
                    g_type = getType(vec[i])

                    if cur_type != g_type:
                        last_pos = i - 1
                    break
                p += 1
        return pro
    def rankCal(rf, psw):
        vec = str2vec(psw)
        last_pos = -1
        pro = calPro(rf, vec, last_pos)

        for i in range(N, 0, -1):
            p = psw_pro[i - 1][1]
            if p > pro:
                return acc_rank[i]
        return acc_rank[0]
    threshold = 10e-7  # 10e-7:1 hour,60000+guess,10e-8:5h,80w guess
    smooth = 0.001
    gram = 6  # gram=3 buxing
    N = 100000
    data = []
    target = []
    rfclass = []  # 分类的数字形式
    classNum = 0
    Guess = queue.Queue()  # tuple:pro
    GuessSort = {}
    psw_set="taobao"
    file = open(train_file, 'r')
    testfile=open(test_file, 'r')
    fout = open(sample_file, "w")
    gout = open(guess_file,'w')
    psw_pro = []
    acc_rank = [1]
    st_vec = [1] * gram
    #########################cal char feature dic
    keyboard_pattern = ["1234567890-=", "qwertyuiop[]\\", "asdfghjkl;\'", "zxcvbnm,./"]
    shift_keyboard_pattern = ["!@#$%^&*()_+", "QWERTYUIOP{}|", "ASDFGHJKL:\"", "ZXCVBNM<>?"]
    kp_dic = {}  # 键盘特征
    for i in range(len(keyboard_pattern)):
        for j in range(len(keyboard_pattern[i])):
            kp_dic[keyboard_pattern[i][j]] = (i + 1, j + 1)
            kp_dic[shift_keyboard_pattern[i][j]] = (i + 1, j + 1)
    no_dic = {1:(1, 0)}  # 序号特征
    spchr = 1
    for i in range(32, 127):  # 可见字符32-126
        if i >= ord('0') and i <= ord('9'):
            no_dic[i] = (0, i - ord('0'))
            if i == ord('0'):
                no_dic[i] = (0, 10)
        elif i >= ord('a') and i <= ord('z'):
            no_dic[i] = (3, i - ord('a') + 1)
        elif i >= ord('A') and i <= ord('Z'):
            no_dic[i] = (2, i - ord('A') + 1)
        else :
            no_dic[i] = (1, spchr)
            spchr += 1
    feature_dic={}
    for k in no_dic:
        if k==1:
            feature_dic[1]=[1,0,0,0]
        else:
            if chr(k) in kp_dic.keys():
                feature_dic[k]=list(no_dic[k]+kp_dic[chr(k)])
            else :
                feature_dic[k]=list(no_dic[k]+(0,0))
    smooth_classNum = 0
    n = 0
    for line in file:
        if n == 10000000:
            break
        psw = line.strip('\r\n')
        psw = ''.join([i if 31<ord(i)<127 else '' for i in psw])
        if len(psw) >0 and len(psw)<31:
            n = n + 1
            vec = str2vec(psw)
            last_pos = -1
            for i in range(gram, vec.__len__()):
                if vec[i]>=127:
                    break
                feature_vec = getFeaturevec(vec[:i], last_pos)

                data.append(feature_vec)
                target.append(vec[i])
                cur_type = getType(vec[i - 1])
                g_type = getType(vec[i])

                if cur_type != g_type:
                    last_pos = i - 1
    rf = RandomForestClassifier(n_estimators=30, max_features=0.8, min_samples_leaf=10, random_state=10, n_jobs=10)
    rf.fit(data, target)
    for k in rf.classes_:
        rfclass.append(k)
    classNum = rfclass.__len__()
    smooth_classNum = 1 + smooth * classNum
    rf.n_jobs=1
    getcontext().prec=50
    data=[]
    target=[]
    for i in range(0, N):
        vec = st_vec[:]
        pro = 1
        last_pos=-1
        while 1:
            c, prob,last_pos = randPick(rf, vec,last_pos)
            vec.append(c)
            pro *= prob
            if c == 0:
                break
        psw_pro.append([vec[gram:-1], pro])
    psw_pro = sorted(psw_pro, key=lambda b:b[1], reverse=True)
    for k in psw_pro:
        for t in k[0]:
            fout.write(chr(t))
        fout.write('\t'+str(k[1])+'\n')
    for i in range(0, N):
        acc_rank.append(acc_rank[i] + 1 / (N * float(psw_pro[i][1])))
    for line in testfile:
        psw = line.strip('\r\n')
        psw = ''.join([i if 31<ord(i)<127 else '' for i in psw])
        gout.write(psw+'\t'+str(rankCal(rf,psw))+'\n')
    return
def crackRateSimulate(guessFile):
    guessNumber2CrackRate = {}  # key=猜测数，value=覆盖率
    fin = open(guessFile, 'r', encoding="UTF-8")
    while 1:
        line = fin.readline()
        if not line:
            break
        guessNumber = float(line.split('\t')[-1])
        log = math.ceil(math.log(guessNumber, 10)) if guessNumber >= 1 else 0
        if log not in guessNumber2CrackRate.keys():
            guessNumber2CrackRate[log] = 0
        guessNumber2CrackRate[log] += 1
    fin.close()
    s = sum(guessNumber2CrackRate.values())
    guessNumber2CrackRateSort = sorted(guessNumber2CrackRate.items(), key=lambda x: x[0])
    guessNumberList = []
    crackRateList = []
    pro = 0
    for i in guessNumber2CrackRateSort:
        guessNumberList.append(i[0])
        pro += i[1] / s
        crackRateList.append(pro)
    for i in crackRateList[:15]:
        print(i,end='\t')
    return {g: c for g, c in zip(guessNumberList, crackRateList)}

def _crack_rate_dataset(dataset):
    train_file = "data/training_and_testing_general/{}-Train.txt".format(dataset)
    test_file = "data/training_and_testing_general/{}-Test.txt".format(dataset)
    pro_file = "result/attack/{}-PCFG-Pro".format(dataset)
    sample_file = "result/attack/{}-PCFG-Sample".format(dataset)
    sample_size = 1000000
    guess_file = "result/attack/{}-PCFG-Guess".format(dataset)
    tag2Segment2Pro, segmentOrder, templates = PCFGTrain(train_file)
    PCFGPro(test_file, pro_file,tag2Segment2Pro, templates)
    PCFGSample(sample_size, sample_file, tag2Segment2Pro, segmentOrder, templates)
    guess(sample_file, pro_file, guess_file)
    return crackRateSimulate(guess_file) # PCFG攻击实验
def crack_rate_datasets():
    dataset2crack_rate = {dataset: _crack_rate_dataset(dataset) for dataset in DATASETS}
    save_pickle(crack_rate_datasets_model.model_path, dataset2crack_rate)
    return
def _crack_rate_category(category):
    datasets = CATEGORY2DATASETS[category]
    crack_rate = {i: 0 for i in range(15)}
    for l in range(15):
        p = [crack_rate_datasets_model.get()[dataset][l] for dataset in datasets]
        q = [DATASET2SIZE[dataset] for dataset in datasets]
        crack_rate[l] = weighted_mean(p, q)
    return crack_rate
def crack_rate_categories():
    category2crack_rate = {category: _crack_rate_category(category) for category in CATEGORIES}
    save_pickle(crack_rate_categories_model.model_path, category2crack_rate)
    for category in CATEGORIES:
        print([category2crack_rate[category][i] for i in range(15)])
    return


def to_percent(temp, position):
    return '%1.0f' % (100 * temp) + '%'
def fig(outfile):
    x = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14]
    financial = [0.0038499069366758377, 0.009698921038500877, 0.021774168141568916, 0.042848700935028565, 0.07438601608368474,
     0.1286947579299946, 0.2028974848526513, 0.2893519672793205, 0.3805901194658653, 0.48064329453417187,
     0.5776888721606958, 0.6687430397236399, 0.7467362115330088, 0.8108662062843528, 0.8613854073452949]
    social = [0.009084165249514919, 0.01784446955979435, 0.03242973214109612, 0.06079684286453877, 0.09613981419967511,
     0.152902092664262, 0.27348193632141693, 0.359372654422055, 0.46677946767683076, 0.6050966343571396,
     0.6849540024250536, 0.7535625531236801, 0.8118647536000728, 0.8552412775669632, 0.8871995019705631]
    email = [0.015325693847947271, 0.030874823059460786, 0.054234466483189005, 0.08685157866020404, 0.13924871211390807,
     0.2234321452205238, 0.31678533004066894, 0.4240755267678715, 0.5346074193414087, 0.6450381052133366,
     0.7393339914279704, 0.8166771388440908, 0.876771391179925, 0.9179301983267889, 0.9465858692984609]
    forum = [0.00820261806939984, 0.017698053456852793, 0.04109765994788627, 0.09610798079289747, 0.16138323596454784,
     0.25607276456641603, 0.35798506710982453, 0.4647613412325937, 0.5664294097705563, 0.6645234159411353,
     0.7513636613618356, 0.8207967577431549, 0.8703520109963934, 0.9064247919795373, 0.9332763907401681]
    content = [0.007391993537054831, 0.018045674184972816, 0.0388168403787188, 0.07828113489368707, 0.1463908619418406,
     0.23469780802952647, 0.32845000446131334, 0.4275833500565476, 0.5435014770651452, 0.6492380173472613,
     0.7400247449884312, 0.8147677524032305, 0.8722219949511462, 0.912516061533268, 0.9398263212332423]

    # Markov
    plt.figure(constrained_layout=True, figsize=[8.5, 5])
    font = {'family': 'helvetica', 'size': 17}
    plt.plot(x, financial, label="Financial", color="#6EAE49", linestyle='--', linewidth=3)
    plt.plot(x, social, label="Social", color="#4F5FA8", linestyle=':', linewidth=3)
    plt.plot(x, email, label="Email", color="#894497", linestyle='--', linewidth=3)
    plt.plot(x, forum, label="Forum", color="#EA5029", linestyle=':', linewidth=3)
    plt.plot(x, content, label="Content", color="#94161F", linestyle='--', linewidth=3)
    # plt.fill_between(x[3:], y_financial[3:], y_general[3:], facecolor = "#D2E3F0",)  # Markov
    plt.legend(fontsize=17,loc='upper left', frameon=False, handlelength=5,labelspacing=1)
    plt.xlabel('Guess Number', font)
    plt.ylabel('Fraction of Cracked Passwords', font)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.xscale('symlog')
    plt.xlim(0.9, 1e14+0.1)
    plt.ylim(-0.02, 0.97) # Markov
    plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], fontsize=17)
    plt.xticks([1, 100, 10000, 1000000, 1e8, 1e10, 1e12, 1e14],fontsize=17)
    plt.minorticks_on()
    plt.tick_params(top='in', right='in', which='minor', direction='in')
    plt.tick_params(top='in', right='in', direction='in')
    plt.savefig(outfile, bbox_inches='tight')
    plt.show()
    return





if __name__ == '__main__':
    # fileList = ['BTC', 'ClixSense', 'LiveAuctioneers', 'LinkedIn', 'Twitter', 'Wishbone', 'Badoo', 'Fling',
    #             'Mate1', 'Rockyou', 'Gmail', 'Hotmail', 'Rootkit', 'Xato', 'Yahoo', 'Gawker', 'YouPorn', 'DatPiff']
    # for file in fileList:
    #     trainFile = "Data2024/attack/" + file + "-Train.txt"
    #     testFile = "Data2024/attack/"+file+"-Test.txt"
    #     sampleFile = "Result2024/attack/"+file+"-RF-Sample.txt"
    #     guessFile = "Result2024/attack/"+file+"-RF-Guess.txt"
    #     RF(trainFile, testFile, sampleFile, guessFile)
    #     crackRateSimulate(guessFile) # RF攻击实验
    fig("RF.pdf")
    # fig("RF.png")
    # fig_detail1("RF_detail1")
    # fig_detail2("RF_detail2")
    # fig_detail3("RF_detail3")
    # chi_squared_test("result/attack/Data_for_Chi_Squared_for_RF.txt")

