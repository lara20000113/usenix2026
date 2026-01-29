import random
import math
import csv
#from sklearn.cluster import AgglomerativeClustering
#from scipy.cluster.hierarchy import dendrogram, linkage
# from matplotlib import pyplot as plt
# import scipy.spatial.distance as dist
# from heapq import *
# from matplotlib.ticker import FuncFormatter
# import matplotlib.ticker as ticker
import copy

class Algorithms:
    def __init__(self, MarkovRank, trainFile, genFile, testFile, proFile, sampleFile, guessFile, genSize, sampleSize,
                 maxLength, tags, external=None, order=True, semantic=False):
        self.MarkovRank = MarkovRank
        self.trainFile = trainFile
        self.genFile = genFile
        self.testFile = testFile
        self.proFile = proFile
        self.sampleFile = sampleFile
        self.guessFile = guessFile
        self.genSize = genSize
        self.sampleSize = sampleSize
        self.maxLength = maxLength
        self.tags = tags
        self.external = external
        self.order = order
        self.semantic = semantic
    # 通用
    def CDF(self,proDic, stringList):
        CDFList = []
        s = 0
        for string in stringList:
            s += proDic[string]
            CDFList.append(s)
        return CDFList
    def crackRateGen(self): # 计算真实生成的猜测下的破解率
        pw2Count = {}
        fin = open(self.testFile, 'r')
        while 1:
            line = fin.readline()
            if not line:
                break
            pw = line[:-1].split('\t')[0]
            c = int(line[:-1].split('\t')[1])
            if pw not in pw2Count.keys():
                pw2Count[pw] = 0
            pw2Count[pw] += c
        fin.close()
        s = sum(pw2Count.values())
        crackNumber = 0
        fin = open(self.genFile, 'r')
        crackRateList = []
        guessNumberList = [1]
        guessNumber = 0
        while 1:
            line = fin.readline()
            if not line:
                break
            pw = line[:-1].split('\t')[0]
            guessNumber += 1
            if pw in pw2Count.keys():
                crackNumber += pw2Count[pw]
            if guessNumber==guessNumberList[-1]:
                crackRateList.append(crackNumber/s)
                guessNumberList.append(10*guessNumberList[-1])
        fin.close()
        return guessNumberList, crackRateList
    def crackRateSimulate(self):
        guessNumber2CrackRate = {}  # key=猜测数，value=覆盖率
        fin = open(self.guessFile, 'r', encoding="UTF-8")
        while 1:
            line = fin.readline()
            if not line:
                break
            guessNumber = float(line.split('\t')[-1])
            log = math.ceil(math.log(guessNumber, 10)) if guessNumber >= 1 else 0
            if log not in guessNumber2CrackRate.keys():
                guessNumber2CrackRate[log] = 0
            guessNumber2CrackRate[log] += 1
        s = sum(guessNumber2CrackRate.values())
        guessNumber2CrackRateSort = sorted(guessNumber2CrackRate.items(), key=lambda x: x[0])
        guessNumberList = []
        crackRateList = []
        pro = 0
        for i in guessNumber2CrackRateSort:
            guessNumberList.append(i[0])
            pro += i[1] / s
            crackRateList.append(pro)
        fin.close()
        return guessNumberList, crackRateList
    def binarySearch(self,list, i):  # 大于等于i的最小索引
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
    def guess(self):  # pw f guessnumber
        samples = []
        fin = open(self.sampleFile, 'r')
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
        fin = open(self.proFile, 'r', encoding="UTF-8")
        fout = open(self.guessFile, 'w', encoding="UTF-8")
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
    # (semantic)PCFG
    def LDSParse(self, string):  # 只分析LDS结构
        string += '\n'
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
    def individualStructureParse(self, line):  # 分割一个口令
        tag2Segments = {}
        if self.semantic:
            dList = []
            lList = []
            sList = []
            template = []  # 口令结构
            semantics = eval(line[:-1].split('\t')[2])
            details = eval(line[:-1].split('\t')[1])
            for detail, semantic in zip(details, semantics): # 遍历details和semantics
                if ':' in semantic:   ###########################
                    semantic = semantic.split(':')[0]
                if semantic in self.tags[:11]:
                    template.append(semantic + '.' + str(len(detail)))  # 口令结构
                    if semantic not in tag2Segments.keys():  # 加入tag2Segments字典
                        tag2Segments[semantic] = []
                    tag2Segments[semantic].append(detail)
                else:
                    dSubList, lSubList, sSubList, subTemplate = self.LDSParse(detail)
                    template += subTemplate
                    dList += dSubList
                    lList += lSubList
                    sList += sSubList
        else:
            dList, lList, sList, template = self.LDSParse(line[:-1].split('\t')[0])
        tag2Segments['D'] = dList
        tag2Segments['L'] = lList
        tag2Segments['S'] = sList
        return tag2Segments, template
    def PCFGTrain(self):
        templates = {}  # 结构统计
        tag2segment2Pro = {tag: {} for tag in self.tags}  # 所有段的概率
        if self.external != None: # 如果引入了外部字典
            for d in self.external.keys():
                if self.external[d] not in tag2segment2Pro.keys():
                    tag2segment2Pro[self.external[d]] = {}
                fin = open(d, 'r', encoding="UTF-8")
                while 1:
                    line = fin.readline()
                    if not line:
                        break
                    string = line[:-1]
                    if len(string) > self.maxLength:
                        continue
                    if string not in tag2segment2Pro[self.external[d]].keys():
                        tag2segment2Pro[self.external[d]][string] = 1
                fin.close()
        fin = open(self.trainFile, 'r', encoding="UTF-8")
        while 1:
            line = fin.readline()
            if not line:
                break
            tag2Segments, template = self.individualStructureParse(line)
            for tag in tag2Segments.keys():
                for segment in tag2Segments[tag]:
                    if segment not in tag2segment2Pro[tag].keys():
                        tag2segment2Pro[tag][segment] = 0
                    tag2segment2Pro[tag][segment] += 1
            templateString = ':'.join(template)
            if templateString not in templates.keys():
                templates[templateString] = 0
            templates[templateString] += 1
        fin.close()
        # 各个段和结构的概率统计和频率排名
        segmentOrder = {tag: {length: [] for length in range(1, self.maxLength+1)} for tag in self.tags}
        for tag in self.tags:
            segmentLengthNum = [0] * self.maxLength
            for string in tag2segment2Pro[tag].keys():
                segmentLengthNum[len(string) - 1] += tag2segment2Pro[tag][string]
            for string in tag2segment2Pro[tag].keys():
                tag2segment2Pro[tag][string] /= segmentLengthNum[len(string) - 1]
            segmentSort = sorted(tag2segment2Pro[tag].items(), key=lambda x: x[1], reverse=True)
            for item in segmentSort:
                string = item[0]
                if len(string)==0:
                    continue
                segmentOrder[tag][len(string)].append(string)
        s = sum(templates.values())
        for template in templates.keys():
            templates[template] /= s
        return tag2segment2Pro, segmentOrder, templates
    def PCFGGuess(self, tag2segment2Pro, segmentOrder, templates):
        def generate(str, template, index, length):  # 生成器生成模板下的所有口令
            if index==length:
                yield str
            else:
                tag = template[index].split('.')[0]
                l = int(template[index].split('.')[1])
                for string in segmentOrder[tag][l]:
                    yield from generate(str+string, template, index+1, length)
        class PCFGNode:
            def __init__(self, structure, detail, index, pivot, pro, n):
                self.structure = structure
                self.detail = detail
                self.index = index
                self.pivot = pivot
                self.pro = pro
                self.n = n
            def __lt__(self, other):
                return self.pro >= other.pro
        fout = open(self.genFile, 'w', encoding="UTF-8")
        if self.order:
            # 初始化
            candidate = []
            for template in templates.keys():
                try:
                    pro = templates[template]
                    structure = template.split(':')
                    detail = []
                    index = []
                    for part in structure:
                        tag = part.split('.')[0]
                        length = int(part.split('.')[1])
                        segment = segmentOrder[tag][length][0]
                        detail.append(segment)
                        index.append(0)
                        pro *= tag2segment2Pro[tag][segment]
                    node = PCFGNode(structure, detail, index, 0, pro, len(detail))
                    heappush(candidate, node)
                except:
                    print(template)
            # 生成猜测
            c = 0
            while c < self.genSize and len(candidate) != 0:
                pop = heappop(candidate)
                fout.write(''.join(pop.detail)+'\n')
                c += 1
                n = pop.n
                pivot = pop.pivot
                while pivot < n:
                    detail = pop.detail[:]  # 被push的口令的detail
                    pro = pop.pro  # 被push的口令的pro
                    index = pop.index[:]  # 被push的口令的index
                    tag = pop.structure[pivot].split('.')[0]  # 当前pivot的结构的标签
                    length = int(pop.structure[pivot].split('.')[1])  # 当前pivot的结构的长度
                    try:
                        index[pivot] += 1
                        string = segmentOrder[tag][length][index[pivot]]
                        pro = pro/tag2segment2Pro[tag][detail[pivot]]*tag2segment2Pro[tag][string]
                        detail[pivot] = string
                        node = PCFGNode(pop.structure, detail, index, pivot, pro, n)
                        heappush(candidate, node)
                    except IndexError:
                        pass
                    pivot += 1
        else:  # 穷举生成
            for template in templates:
                if template.count('D')>1 or template.count('L')>1 or template.count('S')>1:
                    continue
                print(template)
                structure = template.split(':')
                g = generate('', structure, 0, len(structure))
                while 1:
                    try:
                        fout.write(next(g)+'\n')
                    except StopIteration:
                        break
        fout.close()
        return
    def PCFGPro(self,tag2segment2Pro, templates):
        fin = open(self.testFile, 'r')
        fout = open(self.proFile, 'w')
        while 1:
            line = fin.readline()
            if not line:
                break
            # pw = line[:-1] # PCFG
            pw = line[:-1].split('\t')[0] # semantic-PCFG
            tag2Segments, template = self.individualStructureParse(line)
            try:
                pro = templates[':'.join(template)]
                for tag in tag2Segments.keys():
                    for segment in tag2Segments[tag]:
                        pro *= tag2segment2Pro[tag][segment]
            except KeyError:
                pro = 0
            fout.write(pw + '\t' + str(pro) + '\n')
        fin.close()
        fout.close()
        return
    def PCFGSample(self, tag2segment2Pro, segmentOrder, templates):
        tag2Length2CDF = {tag: {length: CDF(tag2segment2Pro[tag], segmentOrder[tag][length]) # PCFG
                                for length in range(1, 31)} for tag in tag2segment2Pro.keys()} # PCFG
        # tag2SegmentsList = {tag: [segment for segment in tag2segment2Pro[tag].keys()] for tag in tag2segment2Pro.keys()}
        # tag2CDF = {tag: CDF(tag2segment2Pro[tag], tag2SegmentsList[tag]) for tag in tag2segment2Pro}
        templatesList = [template for template in templates.keys()]
        templatesCDF = CDF(templates, templatesList)
        # 生成sample
        c = 0
        fout = open(self.sampleFile, 'w')
        while c < self.sampleSize:
            r = random.random()
            index = binarySearch(templatesCDF, r)
            structure = templatesList[index]
            pro = templates[templatesList[index]]  # 结构概率
            parts = structure.split(':') # PCFG
            # parts = structure[1:-1].split('][')
            for part in parts:
                tag = part.split('.')[0] # PCFG
                length = int(part.split('.')[1]) # PCFG
                r = random.random()
                # index = binarySearch(tag2CDF[part], r)
                # pro *= tag2segment2Pro[part][tag2SegmentsList[part][index]]
                index = binarySearch(tag2Length2CDF[tag][length], r) # PCFG
                pro *= tag2segment2Pro[tag][segmentOrder[tag][length][index]] # PCFG
            c += 1
            fout.write(str(pro) + '\n')
        fout.close()
        return

    def PCFG(self):
        tag2segment2Pro, segmentOrder, templates = self.PCFGTrain()
        if self.genFile!=None:
            self.PCFGGuess(tag2segment2Pro, segmentOrder, templates)
        if self.testFile!=None:
            self.PCFGPro(tag2segment2Pro, templates)
        if self.sampleFile!=None:
            self.PCFGSample(tag2segment2Pro, segmentOrder, templates)
        if self.guessFile!=None:
            self.guess()
            guessNumberList, crackRateList = self.crackRateSimulate()
            for i in crackRateList[:15]:
                print(i, end='\t')
            print()
    # (semantic)Markov
    def individualParse(self,line):
        if self.semantic:
            semanticTag2Symbol = {tag: chr(index + 11) for index, tag in enumerate(self.tags)}
            semanticStructure = eval(line[:-1].split('\t')[3])  # 口令的semantic结构
            semanticSegments = eval(line[:-1].split('\t')[2]) # 口令的详细semantic段
            tag2Segments = {}
            pw = '\a'*self.MarkovRank
            for structure, segment in zip(semanticStructure, semanticSegments):
                structure = structure.split(':')[0]
                if structure in self.tags: # 存在语义结构
                    if structure not in tag2Segments.keys():
                        tag2Segments[structure] = []
                    tag2Segments[structure].append(segment) # 记录语义结构和对应的语义段
                    pw += semanticTag2Symbol[structure]
                else:
                    pw += structure
            return tag2Segments, pw+'\n'
        else:
            pw = '\a'*self.MarkovRank + line[:-1].split('\t')[0] + '\n'
            return pw
    def MarkovTrain(self):
        if self.semantic:
            tag2segment2Pro = {tag: {} for tag in self.tags}  # 所有段的概率
        subString = {}
        prefix = {}
        charSet = set([])
        fin = open(self.trainFile,'r',encoding="UTF-8")
        while 1:
            line = fin.readline()
            if not line:
                break
            if self.semantic: # 如果是语义模式，记录语义标签和语义段频数
                tag2Segments, pw = self.individualParse(line)
                for tag in tag2Segments:
                    for segment in tag2Segments[tag]:
                        if segment not in tag2segment2Pro[tag].keys():
                            tag2segment2Pro[tag][segment] = 0
                        tag2segment2Pro[tag][segment] += 1
            else:
                pw = self.individualParse(line)
            for char in pw[self.MarkovRank:]:
                charSet.add(char)
            for index in range(len(pw)-self.MarkovRank):
                front = pw[index:index+self.MarkovRank]
                if front in prefix.keys():
                    prefix[front] += 1
                else:
                    prefix[front] = 1
                if front in subString.keys():
                    if pw[index+self.MarkovRank] in subString[front].keys():
                        subString[front][pw[index+self.MarkovRank]] += 1
                    else:
                        subString[front][pw[index+self.MarkovRank]] = 1
                else:
                    subString[front] = {}
                    subString[front][pw[index+self.MarkovRank]] = 1
        fin.close()
        if self.semantic:
            for tag in tag2segment2Pro.keys():
                s = sum(tag2segment2Pro[tag].values())
                for segment in tag2segment2Pro[tag].keys():
                    tag2segment2Pro[tag][segment] /= s
        if self.semantic:
            return subString, prefix, charSet, tag2segment2Pro
        else:
            return subString, prefix, charSet
    def MarkovPro(self, subString, prefix, charSet, tag2segment2Pro=None):
        l = len(charSet)
        fout = open(self.proFile, 'w', encoding="UTF-8")
        fin = open(self.testFile, 'r', encoding="UTF-8")
        while 1:
            line = fin.readline()
            if not line:
                break
            if self.semantic:
                tag2Segments, pw = self.individualParse(line)
            else:
                pw = self.individualParse(line)
            pwInit = line[:-1]
            p = 1
            for index in range(len(pw) - self.MarkovRank):
                front = pw[index:index + self.MarkovRank]
                if front in prefix.keys():
                    if pw[index + self.MarkovRank] in subString[front].keys():
                        p = p * ((subString[front][pw[index + self.MarkovRank]] + 0.01) / (
                                    prefix[front] + 0.01 * l))
                    else:
                        p = p * (0.01 / (prefix[front] + 0.01 * l))
                else:
                    p = p / l
            if self.semantic:
                for tag in tag2Segments.keys():
                    for segment in tag2Segments[tag]:
                        try:
                            p *= tag2segment2Pro[tag][segment]
                        except:
                            p = 0
            fout.write(pwInit + '\t' + str(p) + '\n')
        fin.close()
        fout.close()
        return
    def MarkovSample(self, subString, prefix, charSet, tag2segment2Pro=None):
        l = len(charSet)
        charSet = list(charSet) # 字符集转化为字符列表
        fout = open(self.sampleFile, 'w', encoding="UTF-8")
        prefixCDF = {}
        for front in subString.keys():
            for c in charSet:
                if c in subString[front].keys():
                    subString[front][c] = (subString[front][c]+0.01)/(prefix[front]+0.01*l)
                else:
                    subString[front][c] = 0.01/(prefix[front]+0.01*l)
            prefixCDF[front] = self.CDF(subString[front], charSet)
        prefixCDF[None] = [(index+1)/l for index, c in enumerate(charSet)] # 加入不存在前缀的特殊情况
        subString[None] = {c: 1/l for c in charSet}
        if self.semantic:
            symbol2SemanticTag = {chr(index + 11): tag for index, tag in enumerate(self.tags)}
            tag2SegmentOrder = {tag: [segment for segment in tag2segment2Pro[tag].keys()] for tag in tag2segment2Pro.keys()}
            segmentCDF = {tag: self.CDF(tag2segment2Pro[tag], tag2SegmentOrder[tag]) for tag in tag2segment2Pro}
        num = 0
        while num < self.sampleSize:
            p = 1
            string = '\a'*self.MarkovRank
            end = None
            while end != '\n':
                front = string[-self.MarkovRank:]
                r = random.random()
                if front not in prefix.keys():
                    front = None
                index = self.binarySearch(prefixCDF[front], r)
                end = charSet[index]
                if self.semantic and end in symbol2SemanticTag.keys():
                    tag = symbol2SemanticTag[end]
                    r = random.random()
                    index = self.binarySearch(segmentCDF[tag], r)
                    p *= tag2segment2Pro[tag][tag2SegmentOrder[tag][index]]
                endPro = subString[front][end]
                p *= endPro
                string += end
            fout.write(str(p)+'\n')
            num += 1
        fout.close()
        return
    def MarkovGuess(self, subString, prefix, charSet, tag2segment2Pro=None):
        global guesses, fout
        def dfs(string, pro, r, l, segments=None):
            global guesses, fout
            pre = string[-self.MarkovRank:]  # 前缀
            if pre not in prefix.keys():  # 前缀不存在
                if pro/num > l:  #
                    for c in charSet:
                        if c != '\n':
                            if self.semantic and c in symbol2SemanticTag.keys(): # 如果是semantic模式
                                tag = symbol2SemanticTag[c]
                                for segment in tag2segment2Pro[tag].keys():
                                    if pro/num * tag2segment2Pro[tag][segment] > l:
                                        dfs(string + c, pro/num * tag2segment2Pro[tag][segment], r, l, segments + [segment])
                            elif self.semantic:
                                dfs(string + c, pro/num, r, l, segments)
                            else:
                                dfs(string+c, pro/num, r, l)
                        elif pro/num <= r:
                            guesses += 1
                            if self.semantic:
                                chars = list(string[self.MarkovRank:])
                                index1 = 0  # segments索引
                                for index, char in enumerate(chars):
                                    if char in symbol2SemanticTag.keys():
                                        chars[index] = segments[index1]
                                        index1 += 1
                                fout.write(''.join(chars)+'\t'+str(pro/num)+'\n')
                            else:
                                fout.write(string[self.MarkovRank:]+'\t'+str(pro/num)+'\n')
            else: # 前缀存在
                for c in charSet:
                    proTmp = pro # 原概率的副本
                    if c in subString[pre].keys():
                        proTmp *= (subString[pre][c]+0.01)/(prefix[pre]+0.01*num)
                    else:
                        proTmp *= 0.01/(prefix[pre]+0.01*num)
                    if proTmp > l:
                        if c != '\n':
                            if self.semantic and c in symbol2SemanticTag.keys(): # 如果是semantic模式
                                tag = symbol2SemanticTag[c]
                                for segment in tag2segment2Pro[tag].keys():
                                    if proTmp*tag2segment2Pro[tag][segment]>l:
                                        dfs(string+c, proTmp*tag2segment2Pro[tag][segment],r,l,segments+[segment])
                            elif self.semantic:
                                dfs(string + c, proTmp, r, l, segments)
                            else:
                                dfs(string+c,proTmp,r,l)
                        elif proTmp <= r:
                            if self.semantic:
                                chars = list(string[self.MarkovRank:])
                                index1 = 0  # segments索引
                                for index, char in enumerate(chars):
                                    if char in symbol2SemanticTag.keys():
                                        chars[index] = segments[index1]
                                        index1 += 1
                                fout.write(''.join(chars) + '\t' + str(proTmp) + '\n')
                            else:
                                fout.write(string[self.MarkovRank:]+'\t'+str(proTmp) + '\n')
            return

        symbol2SemanticTag = {chr(index + 11): tag for index, tag in enumerate(self.tags)}
        num = len(charSet)
        guesses = 0
        fout = open(self.genFile, 'w')
        r = 1
        l = 0.1
        while guesses < self.genSize:
            if self.semantic:
                dfs('\a'*self.MarkovRank, 1, r, l, [])
            else:
                dfs('\a' * self.MarkovRank, 1, r, l)
            r = l
            l /= 10
        fout.close()
        return
    def Markov(self):
        if self.semantic:
            subString, prefix, charSet, tag2segment2Pro = self.MarkovTrain()
            if self.testFile != None:
                self.MarkovPro(subString, prefix, charSet, tag2segment2Pro)
            if self.sampleFile != None:
                self.MarkovSample(subString, prefix, charSet, tag2segment2Pro)
            if self.guessFile != None:
                self.guess()
        else:
            subString, prefix, charSet = self.MarkovTrain()
            if self.genFile != None:
                self.MarkovGuess(subString, prefix, charSet)
            if self.testFile != None:
                self.MarkovPro(subString, prefix, charSet)
            if self.sampleFile != None:
                self.MarkovSample(subString, prefix, charSet)
            if self.guessFile != None:
                self.guess()
                guessNumberList, crackRateList = self.crackRateSimulate()
                for i in crackRateList[:15]:
                    print(i,end='\t')
        return

def CDF(proDic, stringList):
    CDFList = []
    s = 0
    for string in stringList:
        s += proDic[string]
        CDFList.append(s)
    return CDFList

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

def sampleGen(sampleFile, sampleSize, tag2segment2Pro, templates):
    # segmentOrder = None # PCFG
    # tag2Length2CDF = {tag: {length: CDF(tag2segment2Pro[tag], segmentOrder[tag][length]) # PCFG
    #                         for length in range(1, 31)} for tag in tag2segment2Pro.keys()} # PCFG
    tag2SegmentsList = {tag: [segment for segment in tag2segment2Pro[tag].keys()] for tag in tag2segment2Pro.keys()}
    tag2CDF = {tag: CDF(tag2segment2Pro[tag], tag2SegmentsList[tag]) for tag in tag2segment2Pro}
    templatesList = [template for template in templates.keys()]
    templatesCDF = CDF(templates, templatesList)
    # 生成sample
    c = 0
    fout = open(sampleFile, 'w')
    while c < sampleSize:
        r = random.random()
        index = binarySearch(templatesCDF, r)
        structure = templatesList[index]
        pro = templates[templatesList[index]]  # 结构概率
        # parts = structure.split(':') # PCFG
        parts = structure[1:-1].split('][')
        for part in parts:
            # tag = part.split('.')[0] # PCFG
            # length = int(part.split('.')[1]) # PCFG
            r = random.random()
            index = binarySearch(tag2CDF[part], r)
            pro *= tag2segment2Pro[part][tag2SegmentsList[part][index]]
            # index = binarySearch(tag2Length2CDF[tag][length], r) # PCFG
            # pro *= tag2segment2Pro[tag][segmentOrder[tag][length][index]] # PCFG
        c += 1
        fout.write(str(pro) + '\n')
    fout.close()
    return

def proCal(testFiles, proFiles, tag2segment2Pro, templates): # pw f pro
    pw2Count = {}
    fin = open('G:\\程序\\NLP_Password_Crack-master\\data\\Test.txt', 'r')
    while 1:
        line = fin.readline()
        if not line:
            break
        pw2Count[line[:-1].split('\t')[0]] = int(line[:-1].split('\t')[1])
    fin.close()
    for inFile, outFile in zip(testFiles, proFiles):
        fin = open(inFile, 'r')
        fout = open(outFile, 'w')
        while 1:
            line = fin.readline()
            if not line:
                break
            pro = 1
            structure = line[:-1].split('\t')[0]
            if structure in templates.keys():
                pro *= templates[structure]
            else:
                pro *= 0
            segments = eval(line[:-1].split('\t')[1])
            pw = ''
            for segment in segments:
                seg = segment[0]
                pw += seg
                tag = segment[2]
                if tag in tag2segment2Pro.keys() and seg in tag2segment2Pro[tag].keys():
                    pro *= tag2segment2Pro[tag][seg]
                else:
                    pro *= 0
            # pw = line[:-1].split('\t')[0]
            # f = int(line[:-1].split('\t')[1])
            # tag2Segments, template = self.individualStructureParse(line)
            # try:
            #     pro = templates[':'.join(template)]
            #     for tag in tag2Segments.keys():
            #         for segment in tag2Segments[tag]:
            #             pro *= tag2segment2Pro[tag][segment]
            # except KeyError:
            #     pro = 0
            if pw in pw2Count.keys():
                fout.write(pw + '\t' + str(pw2Count[pw]) + '\t' + str(pro) + '\n')
            else:
                print(pw)
        fin.close()
        fout.close()
    return

def guess(sampleFile, proFiles, guessFiles): # pw f guessnumber
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
        C.append(C[index - 1] + 1 / (sampleSize * samplesSorted[index]))
    C = C[::-1]
    samples = samplesSorted[::-1]
    for proFile, guessFile in zip(proFiles, guessFiles):
        fin = open(proFile, 'r', encoding="UTF-8")
        fout = open(guessFile, 'w', encoding="UTF-8")
        while 1:
            try:
                line = fin.readline()
                if not line:
                    break
                f = int(line[:-1].split('\t')[1])
                pro = float(line[:-1].split('\t')[2])
                pw = line[:-1].split('\t')[0]
                index = binarySearch(samples, pro)
                fout.write(pw + '\t' + str(f) + '\t' + str(int(C[index])) + '\n')
            except:
                print(line[:-1])
                continue
        fin.close()
        fout.close()
    return

def targuessIICrackRate(inFile1, inFile2):
    fin1 = open(inFile1, 'r')
    fin2 = open(inFile2, 'r')
    rates = {0: 0, 1: 0, 2: 0, 3: 0}
    end = False
    all = 0
    while not end:
        line = fin1.readline()
        target = line[:-1].split('\t')[2] # 目标口令
        all += 1
        c = 0
        crack = False
        while c<1000:
            line = fin2.readline()
            if not line:
                end = True
                break
            guess = line[:-1].split('\t')[0]
            c += 1
            # print(guess,'\t',target)
            if guess==target and not crack:
                rates[math.ceil(math.log(c,10))] += 1
                crack = True
    fin1.close()
    fin2.close()
    print(all)
    print(rates)
    return

def hashcatCrackRateAndRule(inFile1, inFile2, inFile3):
    pw2Count = {}
    fin = open(inFile1, 'r')
    while 1:
        line = fin.readline()
        if not line:
            break
        pw = line[:-1].split('\t')[0]
        c = int(line[:-1].split('\t')[1])
        pw2Count[pw] = c
    fin.close()
    # rates = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    fin = open(inFile3, 'r')
    lines = fin.readlines()
    fin.close()
    rules = {line[:-1]: 0 for line in lines}
    cracked = []
    fin = open(inFile2, 'r')
    while 1:
        line1 = fin.readline()
        if not line1:
            break
        line2 = fin.readline()
        line3 = fin.readline()
        line4 = fin.readline()
        line5 = fin.readline()
        line6 = fin.readline()
        line7 = fin.readline()
        pw = ':'.join(line3[:-1].split(':')[1:])
        # guess = int(line6[:-1].split(':')[1].split(' ')[0])
        rule = ':'.join(line4[:-1].split(':')[1:])
        if pw in cracked:
            continue
        else:
            cracked.append(pw)
        # rates[math.ceil(math.log(guess,10))] += pw2Count[pw]
        rules[rule] += pw2Count[pw]
    fin.close()
    s = sum(pw2Count.values())
    # rates = {k: rates[k]/s for k in rates.keys()}
    for rule in rules.keys():
        print(rules[rule]/s)
    # print(rates)
    return

def crackRateSimulate(guessFiles):
    guessNumberLists = []
    crackRateLists = []
    for guessFile in guessFiles:
        print(guessFile)
        guessNumber2CrackRate = {} # key=猜测数，value=覆盖率
        fin = open(guessFile, 'r', encoding="UTF-8")
        c = 0
        while 1:
            c += 1
            try:
                line = fin.readline()
                if not line:
                    break
                guessNumber = float(line.split('\t')[-1])
                f = int(line[:-1].split('\t')[1])
                log = math.ceil(math.log(guessNumber, 10)) if guessNumber >= 1 else 0
                if log not in guessNumber2CrackRate.keys():
                    guessNumber2CrackRate[log] = 0
                guessNumber2CrackRate[log] += f
            except:
                print(c)
        s = sum(guessNumber2CrackRate.values())
        guessNumber2CrackRateSort = sorted(guessNumber2CrackRate.items(), key=lambda x: x[0])
        guessNumberList = []
        crackRateList = []
        pro = 0
        for i in guessNumber2CrackRateSort:
            guessNumberList.append(i[0])
            pro += i[1]/s
            crackRateList.append(pro)
        fin.close()
        guessNumberLists.append(guessNumberList)
        crackRateLists.append(crackRateList)
    return guessNumberLists, crackRateLists

def fig(guessNumberLists, crackRateLists, labels, colors, lineStyles, marks, outFile):
    for guessNumberList, crackRateList, label, color in zip(guessNumberLists, crackRateLists, labels, colors):
        plt.plot(guessNumberList, crackRateList, label=label, c=color)
    plt.show()
    return


    # def to_percent(temp, position):
    #     return '%1.0f' % (100 * temp) + '%'
    # font = {'family': 'helvetica', 'size': 20}
    # plt.figure(constrained_layout=True, figsize=[8, 5.5])
    # lineList = []
    # for guessNumberList, crackRateList, label, color, lineStyle, mark in \
    #     zip(guessNumberLists, crackRateLists, labels, colors, lineStyles, marks):
    #     p, = plt.plot(guessNumberList, crackRateList, label=label, c=color, linestyle=lineStyle, marker=mark, linewidth=3)
    #     lineList.append(p)
    # for guessNumberList, crackRateList, label, color in zip(guessNumberLists, crackRateLists, labels, colors):
    #     p, = plt.plot(guessNumberList, crackRateList, label=label, c=color, linewidth=3)
    #     lineList.append(p)
    # plt.xscale('symlog')
    # plt.xlim(1e0, 1e7)
    # plt.ylim(0, 0.5, 0.01)
    # plt.ylabel("Fraction of Cracked password", font)
    # plt.xlabel("Guess number", font)
    # plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    # plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.10))
    # plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    # plt.yticks(fontsize=17)
    # plt.xticks(fontsize=17)

    # a = plt.legend(lineList, labels, fontsize=20, labelspacing=1, loc='upper left', frameon=False, bbox_to_anchor=(0.1,0.8))
    # b = plt.legend(lineList[:2], labels[2:], fontsize=20, labelspacing=1, loc='upper left', frameon=False, bbox_to_anchor=(0, 1))
    # c = plt.legend(lineList[2:4], labels[2:4], fontsize=20, labelspacing=1, loc='lower right', frameon=False,
    #                bbox_to_anchor=(1, 0.23))
    # c = plt.legend(lineList[3:5], labels[3:5], fontsize=16, labelspacing=1, loc='upper left', frameon=False, bbox_to_anchor=(0.47,1))
    # d = plt.legend(lineList[5:6], labels[5:6], fontsize=16, labelspacing=1, loc='lower left', frameon=False, bbox_to_anchor=(0.15, 0.1))
    # c = plt.legend(lineList[:2], labels[:2], fontsize=20, labelspacing=1, loc='lower right', frameon=False, bbox_to_anchor=(0.6, 0.05))
    # b = plt.legend(lineList[4:], labels[4:], fontsize=17, labelspacing=1, loc='lower right', frameon=False, bbox_to_anchor=(1, 0))
    # c = plt.legend(lineList[7:], labels[7:], fontsize=13, loc='lower right', labelspacing=0.8, frameon=False, bbox_to_anchor=(1, 0.15))
    # d = plt.legend(lineList[7:10], labels[7:10], fontsize=13, loc='lower right', labelspacing=0.8, frameon=False,
    #                bbox_to_anchor=(1, 0.13))
    # plt.gca().add_artist(a)
    # plt.gca().add_artist(b)
    # plt.gca().add_artist(c)
    #plt.legend(loc='upper left', fontsize=14, labelspacing=0.8, frameon=False)
    #plt.grid(b="True", ls="--")
    # plt.savefig(outFile, bbox_inches='tight')
    # plt.show()
    #return

if __name__ == '__main__':
    # ['Repeat', 'segmentRepeat', 'Sequencial', 'Palindrome', 'Keyboard', 'Date-YYYYMMDD',
    # 'Date-YYMMDD', 'Date-MMDD', 'Date-YYYY', 'EnglishWord', 'EnglishName',
    # types = ['Financial', 'Social', 'Email', 'Forum', 'Content']
    # for type in types:
    # algorithm = Algorithms(MarkovRank=None,
    #                        trainFile="J:\\国家语料库\\Source\\.cn.txt",
    #                        genFile="./China-PCFG+dictionary.txt",
    #                        testFile=None,
    #                        proFile=None,
    #                        sampleFile=None,
    #                        guessFile=None,
    #                        genSize=1000000000,
    #                        sampleSize=None,
    #                        maxLength=100,
    #                        tags=['L', 'D', 'S'],
    #                        external={"H:\\DICT\\name\\Chinese_Names_Corpus\\Chinese_Names_Corpus（120W）.txt": 'L'},
    #                        order=True)
    # algorithm.PCFG()
    # guessNumberLists, crackRateLists = algorithm.crackRateSimulate()
    # for guessNumberList, crackRateList in zip(guessNumberLists, crackRateLists):
    #     print(guessNumberList, '\t', crackRateList)
    # PCFGSetting2GuessListAndCrackRate = {train+'-'+test: None for test in types for train in types}
    # PCFG_SemanticSetting2GuessListAndCrackRate = {train+'-'+test: None for test in types for train in types}
    # MarkovSetting2GuessListAndCrackRate = {train + '-' + test: None for test in types for train in types}
    # Markov_SemanticSetting2GuessListAndCrackRate = {train + '-' + test: None for test in types for train in types}
    # RandomForestSetting2GuessListAndCrackRate = {train + '-' + test: None for test in types for train in types}
    # fin = open('./result/attack/CrackRate.txt', 'r')
    # lines = fin.readlines()
    # fin.close()
    # for line in lines[1:26]:
    #     setting = line[:-1].split('\t')[0]
    #     guessNumberList = [10**i for i in eval(line[:-1].split('\t')[1])]
    #     crackRateList = eval(line[:-1].split('\t')[2])
    #     PCFG_SemanticSetting2GuessListAndCrackRate[setting] = [guessNumberList, crackRateList]
    # for line in lines[27:52]:
    #     setting = line[:-1].split('\t')[0]
    #     guessNumberList = [10**i for i in eval(line[:-1].split('\t')[1])]
    #     crackRateList = eval(line[:-1].split('\t')[2])
    #     PCFGSetting2GuessListAndCrackRate[setting] = [guessNumberList, crackRateList]
    # for line in lines[53:78]:
    #     setting = line[:-1].split('\t')[0]
    #     guessNumberList = [10**i for i in eval(line[:-1].split('\t')[1])]
    #     crackRateList = eval(line[:-1].split('\t')[2])
    #     Markov_SemanticSetting2GuessListAndCrackRate[setting] = [guessNumberList, crackRateList]
    # for line in lines[79:104]:
    #     setting = line[:-1].split('\t')[0]
    #     guessNumberList = [10**i for i in eval(line[:-1].split('\t')[1])]
    #     crackRateList = eval(line[:-1].split('\t')[2])
    #     MarkovSetting2GuessListAndCrackRate[setting] = [guessNumberList, crackRateList]
    # for line in lines[105:110]:
    #     setting = line[:-1].split('\t')[0]
    #     guessNumberList = [10**i for i in eval(line[:-1].split('\t')[1])]
    #     crackRateList = eval(line[:-1].split('\t')[2])
    #     RandomForestSetting2GuessListAndCrackRate[setting] = [guessNumberList, crackRateList]
    # types = ['Financial', 'Social', 'Email', 'Forum', 'Content']
    # colors = ["#16A539", "#E167B3", "#70BCDB", "#C92A00", "#FF9500", "#16A539", "#E167B3", "#70BCDB", "#C92A00", "#FF9500"]
    # lineStyles = ['-', '-', '-', '-', '-','--', '--', '--', '--', '--', ]
    # labels = ['Train: Financial; Model: Semantic_PCFG', 'Train: Social; Model: Semantic_PCFG',
    #           'Train: Email; Model: Semantic_PCFG', 'Train: Forum; Model: Semantic_PCFG',
    #           'Train: Content; Model: Semantic_PCFG',
    #           'Train: Financial; Model: PCFG', 'Train: Social; Model: PCFG',
    #           'Train: Email; Model: PCFG', 'Train: Forum; Model: PCFG',
    #           'Train: Content; Model: PCFG']
    # labels = ['Financial-Classic', 'Social-Classic',
    #           'Email-Classic', 'Forum-Classic',
    #           'Content-Classic', 'Financial-semantic', 'Social-semantic',
    #           'Email-semantic', 'Forum-semantic',
    #           'Content-semantic']
    # labels = ['Financial', 'Social', 'Email', 'Forum', 'Content']
    # fig(guessNumberLists= [PCFGSetting2GuessListAndCrackRate[train+'-'+train][0] for train in types],
    #               crackRateLists=[PCFGSetting2GuessListAndCrackRate[train+'-'+train][1] for train in types],
    #               labels=labels, colors=colors, lineStyles=lineStyles, marks=[], outFile="PCFG-Big.pdf")
    # fig(guessNumberLists=[PCFG_SemanticSetting2GuessListAndCrackRate[train + '-' + train][0] for train in types],
    #     crackRateLists=[PCFG_SemanticSetting2GuessListAndCrackRate[train + '-' + train][1] for train in types],
    #     labels=labels, colors=colors, lineStyles=lineStyles, marks=[], outFile="PCFG-semantic-Big.png")
    # fig(guessNumberLists=[PCFGSetting2GuessListAndCrackRate[train+'-Financial'][0] for train in types]+
    #                      [PCFG_SemanticSetting2GuessListAndCrackRate[train + '-Financial'][0] for train in types],
    #     crackRateLists=[PCFGSetting2GuessListAndCrackRate[train+'-Financial'][1] for train in types]+
    #                    [PCFG_SemanticSetting2GuessListAndCrackRate[train + '-Financial'][1] for train in types],
    #     labels=labels, colors=colors, lineStyles=lineStyles, marks=[], outFile="PCFG-semantic-Small-TestFinancial.pdf")
    # fig(guessNumberLists= [MarkovSetting2GuessListAndCrackRate[train+'-'+train][0] for train in types],
    #               crackRateLists=[MarkovSetting2GuessListAndCrackRate[train+'-'+train][1] for train in types],
    #               labels=labels, colors=colors, lineStyles=lineStyles, marks=[], outFile="Markov-Big.pdf")

    # fig(guessNumberLists= [RandomForestSetting2GuessListAndCrackRate[train+'-'+train][0] for train in types],
    #               crackRateLists=[RandomForestSetting2GuessListAndCrackRate[train+'-'+train][1] for train in types],
    #               labels=labels, colors=colors, lineStyles=lineStyles, marks=[], outFile="RandomForest-Big.pdf")
    # targuessIICrackRate("./result/Reuse/Fin-Fin-Test.txt", "./result/Reuse/Fin_Fin_output.txt")
    # colors = ["#16A539", "#C92A00", "#16A539", "#C92A00", "#16A539", "#C92A00"]
    # lineStyles = ['-', '-', '--', '--', '-', '-']
    # marks = ['', '', '', '', '*', '*']
    # labels = ['Financial -> Financial', 'General -> General', 'General -> Financial', 'Financial -> General',
    #           'Financial/General -> Financial', 'Financial/General -> General']
    # guessNumberLists = [[1, 10, 100, 1000], [1, 10, 100, 1000], [1, 10, 100, 1000], [1, 10, 100, 1000],
    #                     [1, 10, 100, 1000], [1, 10, 100, 1000]]
    # crackRateLists = [[0.3885, 0.4231, 0.4516, 0.4783], [0.3919, 0.4317, 0.4599, 0.4821],
    #                   [0.3279, 0.3839, 0.4151, 0.4377], [0.3282, 0.3946, 0.4282, 0.4508],
    #                   [0.3582, 0.4132, 0.4334, 0.4508], [0.3600, 0.4132, 0.4441, 0.4665]]
    # outFile = "TarguessII.png"
    # fig(guessNumberLists, crackRateLists, labels, colors, lineStyles, marks, outFile)
    # hashcatCrackRate("./data/Content-TestF-Hashcat.txt", "./result/attack/Guess/Hashcat-Content-Content-Guess.txt")

    # colors = ["#16A539", "#E167B3", "#70BCDB", "#C92A00", "#FF9500"]
    # lineStyles = ['-', '-', '-', '-', '-', '-']
    # marks = ['', '', '', '', '', '']
    # labels = ['Financial', 'Social', 'Email', 'Forum', 'Content']
    # guessNumberLists = [[1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000],
    #                     [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000],
    #                     [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000],
    #                     [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000],
    #                     [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]]
    # crackRateLists = [[0.0001, 0.0001, 0.0001, 0.0022, 0.0119, 0.0485, 0.1351, 0.3062, 0.5070],
    #                   [0.0014, 0.0014, 0.0023, 0.0047, 0.0137, 0.0476, 0.1270, 0.2621, 0.4403],
    #                   [0.0119, 0.0119, 0.0179, 0.0328, 0.0547, 0.1031, 0.2125, 0.3831, 0.6379],
    #                   [0.0001, 0.0001, 0.0002, 0.0085, 0.0378, 0.1128, 0.2475, 0.4288, 0.5500],
    #                   [0.0002, 0.0002, 0.0007, 0.0054, 0.0183, 0.0674, 0.1835, 0.3621, 0.5176]]
    # outFile = "Hashcat.png"
    # fig(guessNumberLists, crackRateLists, labels, colors, lineStyles, marks, outFile)
    # hashcatCrackRateAndRule("./data/Content-TestF-Hashcat.txt",
    #                         "./result/attack/Guess/Hashcat-Content-Content-Guess.txt",
    #                         "C:\\Users\\625-2\\Downloads\\analytic-password-cracking-master\\analytic-password-cracking-master\\Statistics.txt\\rulelists\\best64.rule")
    # fin = open('./result/attack/SemanticProb.txt', 'r')
    # dic = eval(fin.readline()[:-1])
    # fin.close()
    # tag2segment2Pro = {}
    # for tagSegment in dic.keys():
    #     tag = tagSegment.split('->')[0]
    #     segment = tagSegment.split('->')[1]
    #     if tag not in tag2segment2Pro.keys():
    #         tag2segment2Pro[tag] = {}
    #     tag2segment2Pro[tag][segment] = dic[tagSegment]
    # fin = open('./result/attack/BaseProb.txt', 'r')
    # templates = eval(fin.readline()[:-1])
    # fin.close()
    # sampleGen("Veras-Financial-Sample.txt", 1000000, tag2segment2Pro, templates)
    # proCal(["./result/attack/Prob.txt"], ["Veras-Financial-Financial-Pro.txt"], tag2segment2Pro, templates)
    # guess("./result/attack/Sample/Veras-Financial-Sample.txt", ["./result/attack/Pro/Veras-Financial-Financial-Pro.txt"],
    #       ["./result/attack/Guess/Veras-Financial-Financial-Guess.txt"])
    # guessNumberLists, crackRateLists = crackRateSimulate(["./result/attack/Guess/Veras-Financial-Financial-Guess.txt"])
    # print(guessNumberLists)
    # print(crackRateLists)

    # colors = ["#16A539", "#C92A00"]
    # lineStyles = ['', ' ']
    # marks = [' ', ' ']
    # labels = ['Ours', 'Veras et al.\'s']
    # guessNumberLists = [[1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000, 100000000000,
    #                      1000000000000, 10000000000000, 100000000000000],
    #                     [1, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000, 100000000000,
    #                      1000000000000, 100000000000000]]
    # crackRateLists = [[0.0022289396005832555, 0.0032660214825091472, 0.009546736972456038, 0.035756778757533726,
    #                     0.09732137062024962, 0.1980096916482058, 0.3354753904220482, 0.46990463441790287,
    #                     0.5855940850581397, 0.678172340736973, 0.7537236416907126, 0.804409460963094, 0.8401180267333265,
    #                     0.8639545318438299, 0.8801970097434006],
    #                   [0.0001238696890870804, 0.00322061191626409, 0.017094017094017092, 0.0599529295181469,
    #                    0.13179734918865355, 0.22903505512201167, 0.3071968289359594, 0.34621578099838973,
    #                    0.36368140715966807, 0.3688839341013255, 0.3701226309921963, 0.37098971881580584, 0.37098971881580584]]
    # outFile = "CompareSemantic-Small-Financial.png"
    # fig(guessNumberLists, crackRateLists, labels, colors, lineStyles, marks, outFile)

    print()
    #########################################2024.9.12############################################
    fileList = ["BTC", "Clixsense", "LiveAuctioneers", "Rockyou", "Gmail", "Hotmail", "Rootkit", "Yahoo", "YouPorn"]
    for file in fileList:
        algorithm = Algorithms(MarkovRank=3,
                               trainFile="Data2024/"+file+"-train-semantic.txt",
                               genFile=None,
                               testFile="Data2024/"+file+"-test-semantic.txt",
                               proFile="Result2024/"+file+"-semantic-PCFG-pro.txt",
                               sampleFile="Result2024/"+file+"-semantic-PCFG-sample.txt",
                               guessFile="Result2024/"+file+"-semantic-PCFG-guess.txt",
                               genSize=None,
                               sampleSize=1000000,
                               maxLength=100,
                               tags=['Repeat', 'segmentRepeat', 'Sequencial', 'Palindrome', 'Keyboard', 'Date-YYYYMMDD','Date-YYMMDD', 'Date-MMDD', 'Date-YYYY', 'EnglishWord', 'EnglishName', "L", "D", "S"],
                               external=None,
                               order=True,
                               semantic=True)
        algorithm.PCFG()

