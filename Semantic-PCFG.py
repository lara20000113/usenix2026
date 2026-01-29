import random
import math
from matplotlib.ticker import FuncFormatter
from utils import DATASETS, DATASET2SIZE, CATEGORIES, CATEGORY2DATASETS, load_pickle, save_pickle, weighted_mean
import matplotlib.pyplot as plt

# 自定义类
class model_manage:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
    def get(self):
        if not self.model:
            self.model = load_pickle(self.model_path)
        return self.model
crack_rate_datasets_model = model_manage("result/attack/crack_rate_datasets_PCFG-semantic.pkl")
crack_rate_categories_model = model_manage("result/attack/crack_rate_categories_PCFG-semantic.pkl")

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
    for i in crackRateList[:20]:
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
    semantic_tags = ["single_character_repeat", "segment_repeat", "sequence_downup", "keybord", "palindrome", "YYYYMMDD",
                     "YYMMDD", "MMDD", "YYYY", "love_related", "website_related", "English_word", "firstname", "lastname",
                     "fullname"]
    tag2Segments = {}
    dList = []
    lList = []
    sList = []
    template = []  # 口令结构
    semantics = eval(line[:-1].split('\t')[2])
    details = eval(line[:-1].split('\t')[1])
    for detail, semantic in zip(details, semantics):  # 遍历details和semantics
        if semantic in semantic_tags:
            template.append(semantic + '.' + str(len(detail)))  # 口令结构
            if semantic not in tag2Segments.keys():  # 加入tag2Segments字典
                tag2Segments[semantic] = []
            tag2Segments[semantic].append(detail)
        else:
            dSubList, lSubList, sSubList, subTemplate = LDSParse(detail+'\n')
            template += subTemplate
            dList += dSubList
            lList += lSubList
            sList += sSubList
    tag2Segments['D'] = dList
    tag2Segments['L'] = lList
    tag2Segments['S'] = sList
    return tag2Segments, template
def PCFGTrain(trainFile):
    templates = {}  # 结构统计
    tag2Segment2Pro = {tag: {} for tag in ["single_character_repeat", "segment_repeat", "sequence_downup", "keybord",
                                           "palindrome", "YYYYMMDD", "YYMMDD", "MMDD", "YYYY", "love_related",
                                           "website_related", "English_word", "firstname", "lastname", "fullname", 'L',
                                           'D', 'S']}  # 所有段的概率
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
        pw = line[:-1].split('\t')[0]
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
    train_file = "data/training_and_testing_semantic/{}-Train.txt".format(dataset)
    test_file = "data/training_and_testing_semantic/{}-Test.txt".format(dataset)
    pro_file = "result/attack/{}-PCFG-semantic-Pro".format(dataset)
    sample_file = "result/attack/{}-PCFG-semantic-Sample".format(dataset)
    sample_size = 1000000
    guess_file = "result/attack/{}-PCFG-semantic-Guess".format(dataset)
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
    financial = [0.005874475747979073, 0.010773428549732702, 0.01883397071627975, 0.042204009374320485, 0.0900972450731938,
     0.16331004248787245, 0.2711688729535456, 0.3905456513495798, 0.48703143414385425, 0.5783324176940955,
     0.6542802102969114, 0.7122555469101011, 0.7536134015038489, 0.7846862727551527, 0.8027580183728805]
    social = [0.010892233884046042, 0.016036968907014382, 0.028021103236798375, 0.05602236030847526, 0.10455505883046137,
     0.1743904025236369, 0.3215290336453671, 0.4277579667866513, 0.5177287848107825, 0.596666092717703,
     0.6535995808009961, 0.6950447263979829, 0.7215732581618831, 0.7371561176748973, 0.7473389499172174]
    email = [0.019035584071830668, 0.03162748158308372, 0.04963656377063946, 0.08587682790576992, 0.16313058740029118,
     0.27645397229285323, 0.42786384925626925, 0.5405938484961477, 0.6353469197355771, 0.720339785652348,
     0.7779003201220334, 0.8200673939814467, 0.8445676827937282, 0.8584788318350526, 0.8647005713673571]
    forum = [0.01046660521498275, 0.01883541009418687, 0.040698273583457074, 0.10253351047993113, 0.19597816937638982,
     0.31213383088547925, 0.4367607356797302, 0.5590049468229219, 0.6433966080189054, 0.712069610011815,
     0.7634930470591926, 0.7998048337832281, 0.8211394557721065, 0.8350742481861645, 0.841825027680918]
    content = [0.009749014722526025, 0.01859097755888447, 0.03655620280479857, 0.08157739194113398, 0.16650192765029154,
     0.28637021257827916, 0.40300457956420277, 0.5178397304210641, 0.611664047348868, 0.7002875384903684,
     0.7529209348021519, 0.7869309076172927, 0.8055254605295866, 0.8278534447211571, 0.8327620412395887]

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
    plt.ylim(-0.02, 0.90) # Markov
    plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], fontsize=17)
    plt.xticks([1, 100, 10000, 1000000, 1e8, 1e10, 1e12, 1e14],fontsize=17)
    plt.minorticks_on()
    plt.tick_params(top='in', right='in', which='minor', direction='in')
    plt.tick_params(top='in', right='in', direction='in')
    # plt.savefig(outfile, bbox_inches='tight')
    plt.show()
    return

def tmp():
    dic = {dataset: {i: 0.0 for i in range(1, 16)} for dataset in DATASETS}
    fin = open("test.txt", 'r')
    while 1:
        line = fin.readline()
        if not line:
            break
        parts = line[:-1].split('\t')
        assert len(parts) == 16
        if parts[0] in DATASETS:
            print(parts[0])
        else:
            continue
        for index, part in enumerate(parts[1:]):
            dic[parts[0]][index] = float(part)
    fin.close()
    save_pickle(crack_rate_datasets_model.model_path, dic)

if __name__ == '__main__':
    # crack_rate_datasets()
    # crack_rate_categories()
    # fig()



