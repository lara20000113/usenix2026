import math
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from utils import load_pickle, save_pickle, weighted_mean, DATASETS, DATASET2SIZE, CATEGORIES, CATEGORY2DATASETS

# 自定义类
class model_manage:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
    def get(self):
        if not self.model:
            self.model = load_pickle(self.model_path)
        return self.model
crack_rate_datasets_model = model_manage("result/attack/crack_rate_datasets_Markov.pkl")
crack_rate_categories_model = model_manage("result/attack/crack_rate_categories_Markov.pkl")

def CDF(proDic, stringList):
    CDFList = []
    s = 0
    for string in stringList:
        s += proDic[string]
        CDFList.append(s)
    return CDFList
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
def binarySearch(list, i):  # 大于等于i的最小索引
    l = 0
    r = len(list) - 1
    if list[l] >= i:
        return l
    while l != r - 1:
        mid = (l + r) // 2
        if list[mid] > i:
            r = mid
        elif list[mid] < i:
            l = mid
        else:
            return mid
    return r
def guess(sampleFile, proFile, guessFile):  # pw f guessnumber
    samples = []
    fin = open(sampleFile, 'r')
    while 1:
        line = fin.readline()
        if not line:
            break
        samples.append(float(line[:-1]))
    fin.close()
    samplesSorted = sorted(samples, reverse=True)  # sample排序
    sampleSize = len(samples)
    C = [1 / (sampleSize * samplesSorted[0])]  # 求猜测数列表
    for index in range(1, sampleSize):
        if sampleSize * samplesSorted[index] == 0:
            break
        C.append(C[index - 1] + 1 / (sampleSize * samplesSorted[index]))
    C = C[::-1]
    samples = samplesSorted[::-1]
    fin = open(proFile, 'r', encoding="UTF-8")
    fout = open(guessFile, 'w', encoding="UTF-8")
    while 1:
        try:
            line = fin.readline()
            if not line:
                break
            pro = float(line[:-1].split('\t')[1])
            pw = line[:-1].split('\t')[0]
            index = binarySearch(samples, pro)
            fout.write(pw + '\t' + str(int(C[index])) + '\n')
        except:
            print(line[:-1])
            continue
    fin.close()
    fout.close()
    return
def MarkovTrain(trainFile, MarkovRank):
    subString = {}
    prefix = {}
    charSet = set([])
    fin = open(trainFile,'r',encoding="UTF-8")
    while 1:
        line = fin.readline()
        if not line:
            break
        pw = '\a'*MarkovRank + line
        for char in pw[MarkovRank:]:
            charSet.add(char)
        for index in range(len(pw)-MarkovRank):
            front = pw[index:index+MarkovRank]
            if front in prefix.keys():
                prefix[front] += 1
            else:
                prefix[front] = 1
            if front in subString.keys():
                if pw[index+MarkovRank] in subString[front].keys():
                    subString[front][pw[index+MarkovRank]] += 1
                else:
                    subString[front][pw[index+MarkovRank]] = 1
            else:
                subString[front] = {}
                subString[front][pw[index+MarkovRank]] = 1
    fin.close()
    return subString, prefix, charSet
def MarkovPro(proFile, testFile, MarkovRank, subString, prefix, charSet):
    l = len(charSet)
    fout = open(proFile, 'w', encoding="UTF-8")
    fin = open(testFile, 'r', encoding="UTF-8")
    while 1:
        line = fin.readline()
        if not line:
            break
        password = line[:-1]
        pw = '\a'*MarkovRank + line
        p = 1
        for index in range(len(pw) - MarkovRank):
            front = pw[index:index + MarkovRank]
            if front in prefix.keys():
                if pw[index + MarkovRank] in subString[front].keys():
                    p = p * ((subString[front][pw[index + MarkovRank]] + 0.01) / (
                                prefix[front] + 0.01 * l))
                else:
                    p = p * (0.01 / (prefix[front] + 0.01 * l))
            else:
                p = p / l
        fout.write(password + '\t' + str(p) + '\n')
    fin.close()
    fout.close()
    return
def MarkovSample(sampleFile, sampleSize, MarkovRank, subString, prefix, charSet):
    l = len(charSet)
    charSet = list(charSet) # 字符集转化为字符列表
    fout = open(sampleFile, 'w', encoding="UTF-8")
    prefixCDF = {}
    for front in subString.keys():
        for c in charSet:
            if c in subString[front].keys():
                subString[front][c] = (subString[front][c]+0.01)/(prefix[front]+0.01*l)
            else:
                subString[front][c] = 0.01/(prefix[front]+0.01*l)
        prefixCDF[front] = CDF(subString[front], charSet)
    prefixCDF[None] = [(index+1)/l for index, c in enumerate(charSet)] # 加入不存在前缀的特殊情况
    subString[None] = {c: 1/l for c in charSet}
    num = 0
    while num < sampleSize:
        p = 1
        string = '\a'*MarkovRank
        end = None
        while end != '\n':
            front = string[-MarkovRank:]
            r = random.random()
            if front not in prefix.keys():
                front = None
            index = binarySearch(prefixCDF[front], r)
            end = charSet[index]
            endPro = subString[front][end]
            p *= endPro
            string += end
        fout.write(str(p)+'\n')
        num += 1
    fout.close()
    return
def _crack_rate_dataset(dataset):
    train_file = "data/training_and_testing_general/{}-Train.txt".format(dataset)
    test_file = "data/training_and_testing_general/{}-Test.txt".format(dataset)
    pro_file = "result/attack/{}-Markov-Pro.txt".format(dataset)
    sample_file = "result/attack/{}-Markov-Sample.txt".format(dataset)
    sample_size = 1000000
    guess_file = "result/attack/{}-Markov-Guess.txt".format(dataset)
    subString, prefix, charSet = MarkovTrain(train_file, 3)
    MarkovPro(pro_file, test_file, 3, subString, prefix, charSet)
    MarkovSample(sample_file, sample_size, 3, subString, prefix, charSet)
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
def fig():
    x = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14]
    financial = [0.004103118730336682, 0.006487363485562928, 0.012392682370174211, 0.018878489826244793, 0.03405351036692424,
     0.06775885782561304, 0.14416109017947276, 0.25376883528030925, 0.37616111696192994, 0.5010778612241534,
     0.615658858218908, 0.7128315574073044, 0.7887808918985199, 0.8435827128769495, 0.8833603904048418]
    social = [0.0076833746290426545, 0.01163053722383415, 0.016266333067302324, 0.024924113872232163, 0.03997520307709848,
     0.0779614102814333, 0.1870527179526526, 0.32534867908446674, 0.4499403298309788, 0.6218141835559992,
     0.712686942789959, 0.7844096696632881, 0.835000821002968, 0.8708333853727612, 0.8963237880895302]
    email = [0.015555213242974762, 0.020541216486667353, 0.0355825121381579, 0.04862096103813829, 0.08277397100522448,
     0.15745725093685578, 0.27431787085131915, 0.4155199353330623, 0.5606716945437163, 0.6923948062684461,
     0.7890717434566906, 0.862399397290381, 0.9088756551327808, 0.9364688631221656, 0.9545381299249444]
    forum = [0.008160363117301556, 0.012715612770352774, 0.019669632684842637, 0.03327205742409533, 0.07669266278996932,
     0.1651825519784052, 0.30751484140448615, 0.4521538328727873, 0.5851985345753477, 0.6979747433613808,
     0.78813008623577, 0.8514022140446443, 0.8924113272089677, 0.9198903300247471, 0.9392708026466597]
    content = [0.007167970476399009, 0.014576544619251742, 0.022220487166905237, 0.03848764253456583, 0.07712466771454192,
     0.16049418207714614, 0.2707988281162721, 0.4059664504009121, 0.5472436648105502, 0.6791660680262243,
     0.7721595832539099, 0.8395656145966214, 0.8865936434623839, 0.9176137881580885, 0.9389735153669793]

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
    # plt.savefig(outfile, bbox_inches='tight')
    plt.show()
    return




if __name__ == '__main__':
    # crack_rate_datasets()
    # crack_rate_categories()
    # fig()
    dataset2crack_rate = crack_rate_datasets_model.get()
    print([dataset2crack_rate[dataset][7] for dataset in DATASETS])


