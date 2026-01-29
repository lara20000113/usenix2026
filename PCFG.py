import random
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from utils import (DATASETS, DATASET2SIZE, CATEGORIES, CATEGORY2DATASETS,
                   load_pickle, save_pickle, weighted_mean, chi_squared)


# 自定义类
class model_manage:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
    def get(self):
        if not self.model:
            self.model = load_pickle(self.model_path)
        return self.model
crack_rate_datasets_model = model_manage("result/attack/crack_rate_datasets_PCFG.pkl")
crack_rate_categories_model = model_manage("result/attack/crack_rate_categories_PCFG.pkl")

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
def LDSParse(string):  # 只分析LDS结构
    tmpIndex = 0
    dList = []
    lList = []
    sList = []
    dString = ""
    lString = ""
    sString = ""
    template = []
    while string[tmpIndex] != '\n':
        if string[tmpIndex].isdigit():
            while string[tmpIndex].isdigit():
                dString += string[tmpIndex]
                tmpIndex += 1
            dList.append(dString)
            template.append('D.' + str(len(dString)))
            dString = ""
        elif string[tmpIndex].islower() or string[tmpIndex].isupper():
            while string[tmpIndex].islower() or string[tmpIndex].isupper():
                lString += string[tmpIndex]
                tmpIndex += 1
            lList.append(lString)
            template.append('L.' + str(len(lString)))
            lString = ""
        else:
            while not (string[tmpIndex].isupper() or string[tmpIndex].islower() or string[tmpIndex].isdigit() or
                       string[tmpIndex] == '\n'):
                sString += string[tmpIndex]
                tmpIndex += 1
            sList.append(sString)
            template.append('S.' + str(len(sString)))
            sString = ""
    return dList, lList, sList, template
def individualStructureParse(line):
    tag2Segments = {}
    dList, lList, sList, template = LDSParse(line)
    tag2Segments['D'] = dList
    tag2Segments['L'] = lList
    tag2Segments['S'] = sList
    return tag2Segments, template
def PCFGTrain(trainFile):
    templates = {}  # 结构统计
    tag2Segment2Pro = {'L':{}, 'D':{}, 'S':{}}  # 所有段的概率
    fin = open(trainFile, 'r', encoding="UTF-8")
    while 1:
        line = fin.readline()
        if not line:
            break
        tag2Segments, template = individualStructureParse(line)
        for tag in tag2Segments.keys():
            for segment in tag2Segments[tag]:
                if segment not in tag2Segment2Pro[tag].keys():
                    tag2Segment2Pro[tag][segment] = 0
                tag2Segment2Pro[tag][segment] += 1
        templateString = ':'.join(template)
        if templateString not in templates.keys():
            templates[templateString] = 0
        templates[templateString] += 1
    fin.close()
    segmentOrder = {tag: {length: [] for length in range(1, 31)} for tag in tag2Segment2Pro.keys()}
    for tag in tag2Segment2Pro.keys():
        segmentLengthNum = [0] * 30
        for string in tag2Segment2Pro[tag].keys():
            segmentLengthNum[len(string) - 1] += tag2Segment2Pro[tag][string]
        for string in tag2Segment2Pro[tag].keys():
            tag2Segment2Pro[tag][string] /= segmentLengthNum[len(string) - 1]
        segmentSort = sorted(tag2Segment2Pro[tag].items(), key=lambda x: x[1], reverse=True)
        for item in segmentSort:
            string = item[0]
            if len(string)==0:
                continue
            segmentOrder[tag][len(string)].append(string)
    s = sum(templates.values())
    for template in templates.keys():
        templates[template] /= s
    return tag2Segment2Pro, segmentOrder, templates
def PCFGPro(testFile, proFile,tag2Segment2Pro, templates):
    fin = open(testFile, 'r')
    fout = open(proFile, 'w')
    while 1:
        line = fin.readline()
        if not line:
            break
        pw = line[:-1]
        tag2Segments, template = individualStructureParse(line)
        try:
            pro = templates[':'.join(template)]
            for tag in tag2Segments.keys():
                for segment in tag2Segments[tag]:
                    pro *= tag2Segment2Pro[tag][segment]
        except KeyError:
            pro = 0
        fout.write(pw + '\t' + str(pro) + '\n')
    fin.close()
    fout.close()
    return
def PCFGSample(sampleSize, sampleFile, tag2Segment2Pro, segmentOrder, templates):
    tag2Length2CDF = {tag: {length: CDF(tag2Segment2Pro[tag], segmentOrder[tag][length]) # PCFG
                            for length in range(1, 31)} for tag in tag2Segment2Pro.keys()} # PCFG
    templatesList = [template for template in templates.keys()]
    templatesCDF = CDF(templates, templatesList)
    c = 0
    fout = open(sampleFile, 'w')
    while c < sampleSize:
        r = random.random()
        index = binarySearch(templatesCDF, r)
        structure = templatesList[index]
        pro = templates[templatesList[index]]  # 结构概率
        parts = structure.split(':') # PCFG
        for part in parts:
            tag = part.split('.')[0] # PCFG
            length = int(part.split('.')[1]) # PCFG
            r = random.random()
            index = binarySearch(tag2Length2CDF[tag][length], r)
            pro *= tag2Segment2Pro[tag][segmentOrder[tag][length][index]]
        c += 1
        fout.write(str(pro) + '\n')
    fout.close()
    return
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
    return
def to_percent(temp, position):
    return '%1.0f' % (100 * temp) + '%'
def fig():
    x = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14]
    financial = [0.005871227874035071, 0.010481654810475565, 0.01879643675702012, 0.04123618880557256,
                 0.09191875512702911, 0.17487123100656868, 0.28896641089075514, 0.4183489141456081, 0.5061145873776943,
                 0.5759320666308145, 0.6243368225946607, 0.6559970849798475, 0.6777376799416869, 0.6949929816142433,
                 0.706258131160648]
    social = [0.010836254322315594, 0.014774439179959657, 0.02627570771874426, 0.05410876092951372, 0.10693012985443857,
     0.18818985145849498, 0.3473136991952932, 0.43586216700658914, 0.5198322825373957, 0.5679648506203938,
     0.6004727728632419, 0.6201585522517722, 0.6319551673807805, 0.6390349597093051, 0.6432556286639156]
    email = [0.019069941276507737, 0.03231115533932409, 0.0525129430429765, 0.09352883841033695, 0.1770456569018131,
     0.3087458773892044, 0.5001854370540405, 0.61219438005752, 0.6744907201903334, 0.7241185831493215,
     0.7547946707717645, 0.7739182921979172, 0.785993363891633, 0.7935383997135753, 0.7965135896538356]
    forum = [0.010404450733564458, 0.01967125017068146, 0.041591367928372834, 0.1035407559838247, 0.20711864232546365,
     0.34965518947083424, 0.4811117878284407, 0.5923910798988169, 0.6471453183865957, 0.6846772907139719,
     0.7080953538761158, 0.7218572802394355, 0.730573005887327, 0.7365797984692709, 0.7402816021586217]
    content = [0.009729514020369872, 0.01843674608434865, 0.03526021535331031, 0.08328316669921543, 0.17434205325678004,
     0.3126270098810137, 0.42776748824866845, 0.5431590964626557, 0.6154591208248943, 0.6689692930940689,
     0.6933104059030913, 0.706616625290894, 0.7142732985529466, 0.72003023236024, 0.7390433940089327]
    plt.figure(constrained_layout=True, figsize=[8.5, 5])
    font = {'family': 'helvetica', 'size': 17}
    plt.plot(x, financial,label="Financial", color="#6EAE49",linestyle='--', linewidth=3)
    plt.plot(x, social, label="Social", color="#4F5FA8",linestyle=':',linewidth=3)
    plt.plot(x, email, label="Email", color="#894497", linestyle='--', linewidth=3)
    plt.plot(x, forum, label="Forum", color="#EA5029", linestyle=':', linewidth=3)
    plt.plot(x, content, label="Content", color="#94161F", linestyle='--', linewidth=3)
    plt.legend(fontsize=17,loc='upper left', frameon=False, handlelength=5,labelspacing=1)
    plt.xlabel('Guess Number', font)
    plt.ylabel('Fraction of Cracked Passwords', font)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.xscale('symlog')
    plt.xlim(0.9, 1e14+0.1)
    plt.ylim(-0.02, 0.85) # PCFG
    plt.yticks(fontsize=17)
    plt.xticks([1, 100, 10000, 1000000, 1e8, 1e10, 1e12, 1e14],fontsize=17)
    plt.minorticks_on()
    plt.tick_params(top='in', right='in', which='minor', direction='in')
    plt.tick_params(top='in', right='in', direction='in')
    #plt.savefig(outfile, bbox_inches='tight')
    plt.show()
    return




if __name__ == '__main__':
    # crack_rate_datasets()
    # crack_rate_categories()
    # fig()
    chi_squared(crack_rate_datasets_model.get())
