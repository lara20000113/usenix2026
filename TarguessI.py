import operator
from collections import OrderedDict
from heapq import *
import math
from utils import (DATASETS_PII, CATEGORIES_PII, CATEGORY2DATASETS_PII, DATASET2SIZE,
                   load_pickle, save_pickle, weighted_mean, chi_squared)
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy import stats

# 自定义类
class model_manage:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
    def get(self):
        if not self.model:
            self.model = load_pickle(self.model_path)
        return self.model
crack_rate_datasets_model = model_manage("result/attack/crack_rate_datasets_TarguessI.pkl")
crack_rate_categories_model = model_manage("result/attack/crack_rate_categories_TarguessI.pkl")

class PI:
    def __init__(self,email,pw,name,gid,account,phone,birth):
        self.email, self.pw, self.name, self.gid, self.account, self.phone = email, pw, name, gid, account,phone
        self.birth = birth
        self.pattern = ""
        self.dStringList = []
        self.lStringList = []
        self.sStringList = []
        self.emailPattern1, self.emailPattern2, self.emailPattern3 = "", "", ""
        self.gidPattern1, self.gidPattern2 = "", ""
        (self.namePattern1, self.namePattern2, self.namePattern3, self.namePattern4, self.namePattern5, self.namePattern6,
         self.namePattern7, self.namePattern8) = "", "", "", "", "", "", "", ""
        self.accountPattern1, self.accountPattern2, self.accountPattern3 = "", "", ""
        (self.birthPattern1, self.birthPattern2, self.birthPattern3, self.birthPattern4, self.birthPattern5,
         self.birthPattern6, self.birthPattern7, self.birthPattern8, self.birthPattern9, self.birthPattern10,
         self.birthPattern11, birthPattern12) = "", "", "", "", "", "", "", "", "", "", "", ""
        self.phonePattern1, self.phonePattern2 = "", ""
        self.infoMap = {}
    def FirstDigitString(self,string):
        string, tmpIndex, dString = string + '\n', 0, ""
        while string[tmpIndex] != '\n':
            if string[tmpIndex].isdigit():
                while string[tmpIndex].isdigit():
                    dString, tmpIndex = dString + string[tmpIndex], tmpIndex + 1
                return dString
            tmpIndex += 1
        return ""
    def FitstLetterString(self,string):
        string, tmpIndex, lString = string + '\n', 0, ""
        while string[tmpIndex] != '\n':
            if string[tmpIndex].isalpha():
                while string[tmpIndex].isalpha():
                    lString, tmpIndex = lString + string[tmpIndex], tmpIndex + 1
                return lString
            tmpIndex += 1
        return ""
    def Valid(self,pos,lenth):
        for index in range(pos,pos+lenth):
            if self.pattern[index] != 'P':
                return False
        return True
    def Convert(self):
        if self.email:
            self.emailPattern1 = self.email.split('@')[0]
            self.emailPattern2 = self.FitstLetterString(self.emailPattern1)
            self.emailPattern3 = self.FirstDigitString(self.emailPattern1)
        if self.gid:
            self.gidPattern1, self.gidPattern2 = self.gid, self.gid[-4:]
        if self.name:
            namePart = self.name.split(' ')
            self.namePattern1, self.namePattern2 = self.name.replace(' ', ''), ''.join(namePart[1:]) + namePart[0]
            self.namePattern3, self.namePattern4 = ''.join([p[0] for p in namePart]), ''.join(namePart[1:])
            self.namePattern5, self.namePattern6 = namePart[0], namePart[0].capitalize()
            self.namePattern7 = ''.join([p[0] for p in namePart[1:]]) + namePart[0]
            self.namePattern8 = namePart[0] + ''.join([p[0] for p in namePart[1:]])
        if self.account:
            self.accountPattern1 = self.account
            self.accountPattern2 = self.FitstLetterString(self.accountPattern1)
            self.accountPattern3 = self.FirstDigitString(self.accountPattern1)
        if self.birth:
            self.birthPattern1, self.birthPattern2 = self.birth, self.birth[:4] + self.birth[5:]
            self.birthPattern3 = self.birth[:6] + self.birth[-1]
            self.birthPattern4 = self.birth[:4] + self.birth[5] + self.birth[7]
            self.birthPattern5, self.birthPattern6, self.birthPattern7 = self.birth[:6], self.birth[-6:], self.birth[:4]
            self.birthPattern8, self.birthPattern9 = self.birth[-4:], self.birth[2:6]
            self.birthPattern10, self.birthPattern11 = self.birth[-4:] + self.birth[:4], self.birth[4:6] + self.birth[:4]
            self.birthPattern12 = self.birth[-4:] + self.birth[2:4]
        if self.phone:
            self.phonePattern1, self.phonePattern2 = self.phone, self.phone[-4:]
        self.infoMap["E1"], self.infoMap["E2"] = self.emailPattern1, self.emailPattern2
        self.infoMap["E3"] = self.emailPattern3
        self.infoMap["G1"], self.infoMap["G2"] = self.gidPattern1, self.gidPattern2
        self.infoMap["N1"], self.infoMap["N2"] = self.namePattern1, self.namePattern2
        self.infoMap["N3"], self.infoMap["N4"] = self.namePattern3, self.namePattern4
        self.infoMap["N5"], self.infoMap["N6"] = self.namePattern5, self.namePattern6
        self.infoMap["N7"], self.infoMap["N8"] = self.namePattern7, self.namePattern8
        self.infoMap["A1"], self.infoMap["A2"] = self.accountPattern1, self.accountPattern2
        self.infoMap["A3"] = self.accountPattern3
        self.infoMap["B1"], self.infoMap["B2"] = self.birthPattern1, self.birthPattern2
        self.infoMap["B3"], self.infoMap["B4"] = self.birthPattern3, self.birthPattern4
        self.infoMap["B5"], self.infoMap["B6"] = self.birthPattern5, self.birthPattern6
        self.infoMap["B7"], self.infoMap["B8"] = self.birthPattern7, self.birthPattern8
        self.infoMap["B9"], self.infoMap["B10"] = self.birthPattern9, self.birthPattern10
        self.infoMap["B11"], self.infoMap["B12"] = self.birthPattern11, self.birthPattern12
        self.infoMap["T1"], self.infoMap["T2"] = self.phonePattern1, self.phonePattern2
    def ProcessEmail(self):
        if self.emailPattern1 != "":
            index, length = self.pw.find(self.emailPattern1), len(self.emailPattern1)
            if index != -1 and self.Valid(index,length):
                self.pattern = self.pattern[:index] + 'E1' + self.pattern[index+length:]
                self.pw = self.pw.replace(self.emailPattern1,"E1", 1)
        if self.emailPattern2 != "":
            index, length = self.pw.find(self.emailPattern2), len(self.emailPattern2)
            if length >= 2 and index != -1 and self.Valid(index,length):
                self.pattern = self.pattern[:index] + 'E2' + self.pattern[index+length:]
                self.pw = self.pw.replace(self.emailPattern2, "E2", 1)
        if self.emailPattern3 != "":
            index, length = self.pw.find(self.emailPattern3), len(self.emailPattern3)
            if length >= 2 and index != -1 and self.Valid(index,length):
                self.pattern = self.pattern[:index] + 'E3' + self.pattern[index+length:]
                self.pw = self.pw.replace(self.emailPattern3, "E3", 1)
    def ProcessGid(self):
        if self.gidPattern1 != "":
            index, length = self.pw.find(self.gidPattern1), len(self.gidPattern1)
            if index != -1 and self.Valid(index,length):
                self.pattern = self.pattern[:index] + 'G1' + self.pattern[index+length:]
                self.pw = self.pw.replace(self.gidPattern1,'G1', 1)
        if self.gidPattern2 != "":
            index, length = self.pw.find(self.gidPattern2), len(self.gidPattern2)
            if index != -1 and self.Valid(index, length):
                self.pattern = self.pattern[:index] + 'G2' + self.pattern[index + length:]
                self.pw = self.pw.replace(self.gidPattern2, 'G2', 1)
    def ProcessName(self):
        if self.namePattern1 != "":
            index, length = self.pw.find(self.namePattern1), len(self.namePattern1)
            if index != -1 and self.Valid(index, length):
                self.pattern = self.pattern[:index] + 'N1' + self.pattern[index + length:]
                self.pw = self.pw.replace(self.namePattern1, 'N1', 1)
        if self.namePattern2 != "":
            index, length = self.pw.find(self.namePattern2), len(self.namePattern2)
            if index != -1 and self.Valid(index, length):
                self.pattern = self.pattern[:index] + 'N2' + self.pattern[index + length:]
                self.pw = self.pw.replace(self.namePattern2, 'N2', 1)
        if self.namePattern3 != "":
            index, length = self.pw.find(self.namePattern3), len(self.namePattern3)
            if index != -1 and self.Valid(index, length):
                self.pattern = self.pattern[:index] + 'N3' + self.pattern[index + length:]
                self.pw = self.pw.replace(self.namePattern3, 'N3', 1)
        if self.namePattern4 != "":
            index, length = self.pw.find(self.namePattern4), len(self.namePattern4)
            if index != -1 and self.Valid(index, length):
                self.pattern = self.pattern[:index] + 'N4' + self.pattern[index + length:]
                self.pw = self.pw.replace(self.namePattern4, 'N4', 1)
        if self.namePattern5 != "":
            index, length = self.pw.find(self.namePattern5), len(self.namePattern5)
            if index != -1 and self.Valid(index, length):
                self.pattern = self.pattern[:index] + 'N5' + self.pattern[index + length:]
                self.pw = self.pw.replace(self.namePattern5, 'N5', 1)
        if self.namePattern6 != "":
            index, length = self.pw.find(self.namePattern6), len(self.namePattern6)
            if index != -1 and self.Valid(index, length):
                self.pattern = self.pattern[:index] + 'N6' + self.pattern[index + length:]
                self.pw = self.pw.replace(self.namePattern6, 'N6', 1)
        if self.namePattern7 != "":
            index, length = self.pw.find(self.namePattern7), len(self.namePattern7)
            if index != -1 and self.Valid(index, length):
                self.pattern = self.pattern[:index] + 'N7' + self.pattern[index + length:]
                self.pw = self.pw.replace(self.namePattern7, 'N7', 1)
        if self.namePattern8 != "":
            index, length = self.pw.find(self.namePattern8), len(self.namePattern8)
            if index != -1 and self.Valid(index, length):
                self.pattern = self.pattern[:index] + 'N8' + self.pattern[index + length:]
                self.pw = self.pw.replace(self.namePattern8, 'N8', 1)
    def ProcessAccount(self):
        if self.accountPattern1 != "":
            index, length = self.pw.find(self.accountPattern1), len(self.accountPattern1)
            if index != -1 and self.Valid(index, length):
                self.pattern = self.pattern[:index] + 'A1' + self.pattern[index + length:]
                self.pw = self.pw.replace(self.accountPattern1, 'A1', 1)
        if self.accountPattern2 != "":
            index, length = self.pw.find(self.accountPattern2), len(self.accountPattern2)
            if length >= 2 and index != -1 and self.Valid(index, length):
                self.pattern = self.pattern[:index] + 'A2' + self.pattern[index + length:]
                self.pw = self.pw.replace(self.accountPattern2, "A2", 1)
        if self.accountPattern3 != "":
            index, length = self.pw.find(self.accountPattern3), len(self.accountPattern3)
            if length >= 2 and index != -1 and self.Valid(index, length):
                self.pattern = self.pattern[:index] + 'A3' + self.pattern[index + length:]
                self.pw = self.pw.replace(self.accountPattern3, "A3", 1)
    def ProcessBirth(self):
        if self.birthPattern1 != "":
            index, length = self.pw.find(self.birthPattern1), len(self.birthPattern1)
            if index != -1 and self.Valid(index, length):
                self.pattern = self.pattern[:index] + 'B1' + self.pattern[index + length:]
                self.pw = self.pw.replace(self.birthPattern1, 'B1', 1)
        if self.birthPattern2 != "":
            index, length = self.pw.find(self.birthPattern2), len(self.birthPattern2)
            if index != -1 and self.Valid(index, length):
                self.pattern = self.pattern[:index] + 'B2' + self.pattern[index + length:]
                self.pw = self.pw.replace(self.birthPattern2, 'B2', 1)
        if self.birthPattern3 != "":
            index, length = self.pw.find(self.birthPattern3), len(self.birthPattern3)
            if index != -1 and self.Valid(index, length):
                self.pattern = self.pattern[:index] + 'B3' + self.pattern[index + length:]
                self.pw = self.pw.replace(self.birthPattern3, 'B3', 1)
        if self.birthPattern4 != "":
            index, length = self.pw.find(self.birthPattern4), len(self.birthPattern4)
            if index != -1 and self.Valid(index, length):
                self.pattern = self.pattern[:index] + 'B4' + self.pattern[index + length:]
                self.pw = self.pw.replace(self.birthPattern4, 'B4', 1)
        if self.birthPattern5 != "":
            index, length = self.pw.find(self.birthPattern5), len(self.birthPattern5)
            if index != -1 and self.Valid(index, length):
                self.pattern = self.pattern[:index] + 'B5' + self.pattern[index + length:]
                self.pw = self.pw.replace(self.birthPattern5, 'B5', 1)
        if self.birthPattern6 != "":
            index, length = self.pw.find(self.birthPattern6), len(self.birthPattern6)
            if index != -1 and self.Valid(index, length):
                self.pattern = self.pattern[:index] + 'B6' + self.pattern[index + length:]
                self.pw = self.pw.replace(self.birthPattern6, 'B6', 1)
        if self.birthPattern7 != "":
            index, length = self.pw.find(self.birthPattern7), len(self.birthPattern7)
            if index != -1 and self.Valid(index, length):
                self.pattern = self.pattern[:index] + 'B7' + self.pattern[index + length:]
                self.pw = self.pw.replace(self.birthPattern7, 'B7', 1)
        if self.birthPattern8 != "":
            index, length = self.pw.find(self.birthPattern8), len(self.birthPattern8)
            if index != -1 and self.Valid(index, length):
                self.pattern = self.pattern[:index] + 'B8' + self.pattern[index + length:]
                self.pw = self.pw.replace(self.birthPattern8, 'B8', 1)
        if self.birthPattern9 != "":
            index, length = self.pw.find(self.birthPattern9), len(self.birthPattern9)
            if index != -1 and self.Valid(index, length):
                self.pattern = self.pattern[:index] + 'B9' + self.pattern[index + length:]
                self.pw = self.pw.replace(self.birthPattern9, 'B9', 1)
        if self.birthPattern10 != "":
            index, length = self.pw.find(self.birthPattern10), len(self.birthPattern10)
            if index != -1 and self.Valid(index, length):
                self.pattern = self.pattern[:index] + 'B10' + self.pattern[index + length:]
                self.pw = self.pw.replace(self.birthPattern10, 'B10', 1)
        if self.birthPattern11 != "":
            index, length = self.pw.find(self.birthPattern11), len(self.birthPattern11)
            if index != -1 and self.Valid(index, length):
                self.pattern = self.pattern[:index] + 'B11' + self.pattern[index + length:]
                self.pw = self.pw.replace(self.birthPattern11, 'B11', 1)
        if self.birthPattern12 != "":
            index, length = self.pw.find(self.birthPattern12), len(self.birthPattern12)
            if index != -1 and self.Valid(index, length):
                self.pattern = self.pattern[:index] + 'B12' + self.pattern[index + length:]
                self.pw = self.pw.replace(self.birthPattern12, 'B12', 1)
    def ProcessPhone(self):
        if self.phonePattern1 != "":
            index, length = self.pw.find(self.phonePattern1), len(self.phonePattern1)
            if index != -1 and self.Valid(index, length):
                self.pattern = self.pattern[:index] + 'T1' + self.pattern[index + length:]
                self.pw = self.pw.replace(self.phonePattern1, 'T1', 1)
        if self.phonePattern2 != "":
            index, length = self.pw.find(self.phonePattern2), len(self.phonePattern2)
            if index != -1 and self.Valid(index, length):
                self.pattern = self.pattern[:index] + 'T2' + self.pattern[index + length:]
                self.pw = self.pw.replace(self.phonePattern2, 'T2', 1)
    def LDSParse(self):
        assert len(self.pattern) == len(self.pw)
        tmpIndex, tmpPattern, self.pattern = 0, "", self.pattern + '\n'
        while self.pattern[tmpIndex] != '\n':
            if self.pattern[tmpIndex] == 'P' and self.pw[tmpIndex].isdigit():
                dString = ""
                while self.pattern[tmpIndex] == 'P' and self.pw[tmpIndex].isdigit():
                    dString, tmpIndex = dString + self.pw[tmpIndex], tmpIndex + 1
                self.dStringList.append(dString)
                tmpPattern = tmpPattern + 'D'+ str(len(dString))
            elif self.pattern[tmpIndex] == 'P' and self.pw[tmpIndex].isalpha():
                lString = ""
                while self.pattern[tmpIndex] == 'P' and self.pw[tmpIndex].isalpha():
                    lString, tmpIndex = lString + self.pw[tmpIndex], tmpIndex + 1
                self.lStringList.append(lString)
                tmpPattern = tmpPattern + 'L' + str(len(lString))
            elif self.pattern[tmpIndex] == 'P' and not (self.pw[tmpIndex].isalpha() or self.pw[tmpIndex].isdigit() or self.pw[tmpIndex] == '\n'):
                sString = ""
                while self.pattern[tmpIndex] == 'P' and not (self.pw[tmpIndex].isalpha() or self.pw[tmpIndex].isdigit() or self.pw[tmpIndex] == '\n'):
                    sString, tmpIndex = sString + self.pw[tmpIndex], tmpIndex + 1
                self.sStringList.append(sString)
                tmpPattern = tmpPattern + 'S' + str(len(sString))
            else:
                tmpPattern, tmpIndex = tmpPattern + self.pattern[tmpIndex], tmpIndex + 1
        self.pattern = tmpPattern
    def Parse(self):
        self.pattern = 'P'*len(self.pw)
        self.Convert()
        if self.name:
            self.ProcessName()
        if self.account:
            self.ProcessAccount()
        if self.birth:
            self.ProcessBirth()
        if self.phone:
            self.ProcessPhone()
        if self.email:
            self.ProcessEmail()
        if self.gid:
            self.ProcessGid()
        self.LDSParse()
maxLength = 31
def Train(trainFile):
    # Variables Definition
    digitSegment, digitSegmentPro, symbolSegment, symbolSegmentPro, letterSegment= {}, {}, {}, {}, {}
    letterSegmentPro, templates, templatesPro = {}, {}, {}
    fin = open(trainFile,'r',encoding="UTF-8")
    while 1:
        line = fin.readline()
        if not line:
            break
        info = line[:-1].split('\t')
        email, pw, username, name, birthday = info[0], info[1], info[2], info[3], info[4].replace('-', '')
        pi = PI(email,pw,name,None,username,None,birthday)
        pi.Parse()
        if pi.pattern in templates.keys():
            templates[pi.pattern] += 1
        else:
            templates[pi.pattern] = 1
        for dString in pi.dStringList:
            if dString in digitSegment.keys():
                digitSegment[dString] += 1
            else:
                digitSegment[dString] = 1
        for lString in pi.lStringList:
            if lString in letterSegment.keys():
                letterSegment[lString] += 1
            else:
                letterSegment[lString] = 1
        for sString in pi.sStringList:
            if sString in symbolSegment.keys():
                symbolSegment[sString] += 1
            else:
                symbolSegment[sString] = 1
    fin.close()
    digitLengthNum = [0] * maxLength
    for dString in digitSegment.keys():
        digitLengthNum[len(dString) - 1] += digitSegment[dString]
    for dString in digitSegment.keys():
        digitSegmentPro[dString] = digitSegment[dString] / digitLengthNum[len(dString) - 1]
    symbolLengthNum = [0] * maxLength
    for sString in symbolSegment.keys():
        symbolLengthNum[len(sString) - 1] += symbolSegment[sString]
    for sString in symbolSegment.keys():
        symbolSegmentPro[sString] = symbolSegment[sString] / symbolLengthNum[len(sString) - 1]
    letterLengthNum = [0] * maxLength
    for lString in letterSegment.keys():
        letterLengthNum[len(lString) - 1] += letterSegment[lString]
    for lString in letterSegment.keys():
        letterSegmentPro[lString] = letterSegment[lString] / letterLengthNum[len(lString) - 1]
    s = sum(templates.values())
    for template in templates.keys():
        templatesPro[template] = templates[template] / s
    digitAllOrder = list(OrderedDict(sorted(digitSegmentPro.items(), key=operator.itemgetter(1), reverse=True)))
    letterAllOrder = list(OrderedDict(sorted(letterSegmentPro.items(), key=operator.itemgetter(1), reverse=True)))
    symbolAllOrder = list(OrderedDict(sorted(symbolSegmentPro.items(), key=operator.itemgetter(1), reverse=True)))
    digitLengthOrder = []
    for i in range(maxLength):
        digitLengthOrder.append([])
    for dString in digitAllOrder:
        digitLengthOrder[len(dString) - 1].append(dString)
    letterLengthOrder = []
    for i in range(maxLength):
        letterLengthOrder.append([])
    for lString in letterAllOrder:
        letterLengthOrder[len(lString) - 1].append(lString)
    symbolLengthOrder = []
    for i in range(maxLength):
        symbolLengthOrder.append([])
    for sString in symbolAllOrder:
        symbolLengthOrder[len(sString) - 1].append(sString)
    return (digitSegmentPro,letterSegmentPro,symbolSegmentPro,templatesPro,digitLengthOrder,letterLengthOrder,
            symbolLengthOrder)
def Parting(baseStructure):
    # Variables Definition
    baseStructure, structure, tmpIndex = baseStructure + '\n', [], 0
    while baseStructure[tmpIndex].isupper():
        string, tmpIndex = baseStructure[tmpIndex], tmpIndex + 1
        while baseStructure[tmpIndex].isdigit():
            string, tmpIndex = string + baseStructure[tmpIndex], tmpIndex + 1
        structure.append(string)
    return structure
class Node:
    def __init__(self,structure,detail,index,pivot,pro):
        self.structure = structure
        self.detail = detail
        self.index = index
        self.pivot = pivot
        self.pro = pro
    def __lt__(self, other):
        return self.pro >= other.pro
def MiddleGuess(digitSegmentPro,letterSegmentPro,symbolSegmentPro,templatesPro,digitLengthOrder,letterLengthOrder,symbolLengthOrder,num,genFile):
    candidate = []
    for template in templatesPro.keys():
        pro, structure, detail, index = templatesPro[template], Parting(template), [], []
        for part in structure:
            if part[0] == 'D':
                dString = digitLengthOrder[int(part[1:]) - 1][0]
                detail.append(dString)
                pro *= digitSegmentPro[dString]
                index.append(0)
            elif part[0] == 'L':
                lString = letterLengthOrder[int(part[1:]) - 1][0]
                detail.append(lString)
                pro *= letterSegmentPro[lString]
                index.append(0)
            elif part[0] == 'S':
                sString = symbolLengthOrder[int(part[1:]) - 1][0]
                detail.append(sString)
                pro *= symbolSegmentPro[sString]
                index.append(0)
            else:
                detail.append(part)
                index.append(0)
        node = Node(structure, detail, index, 0, pro)
        heappush(candidate, node)
    fout = open(genFile, 'w', encoding="UTF-8")
    c = 0
    while c < num and len(candidate) != 0:
        pop, c = heappop(candidate), c + 1
        n, pivot = len(pop.index), pop.pivot
        fout.write(str(pop.structure)+'\t'+str(pop.detail) + '\n')
        while pivot < n:
            g, detailMore, proMore, indexMore = False, pop.detail[:], pop.pro, pop.index[:]
            length, index = int(pop.structure[pivot][1:]), pop.index[pivot]
            if pop.structure[pivot][0] == 'D':
                if index < len(digitLengthOrder[length - 1]) - 1:
                    index = index + 1
                    indexMore[pivot], dString = index, digitLengthOrder[length - 1][index]
                    proMore = proMore / digitSegmentPro[pop.detail[pivot]] * digitSegmentPro[dString]
                    detailMore[pivot], g = dString, True
            elif pop.structure[pivot][0] == 'L':
                if index < len(letterLengthOrder[length - 1]) - 1:
                    index += 1
                    indexMore[pivot], lString = index, letterLengthOrder[length - 1][index]
                    proMore = proMore / letterSegmentPro[pop.detail[pivot]] * letterSegmentPro[lString]
                    detailMore[pivot], g = lString, True
            elif pop.structure[pivot][0] == 'S':
                if index < len(symbolLengthOrder[length - 1]) - 1:
                    index += 1
                    indexMore[pivot], sString = index, symbolLengthOrder[length - 1][index]
                    proMore = proMore / symbolSegmentPro[pop.detail[pivot]] * symbolSegmentPro[sString]
                    detailMore[pivot], g = sString, True
            if g:
                node = Node(pop.structure, detailMore, indexMore, pivot, proMore)
                heappush(candidate, node)
            pivot += 1
    fout.close()
def guess(template_file, test_file):
    fin = open(template_file, 'r', encoding="utf-8")
    lines = fin.readlines()
    fin.close()
    templates = [[eval(line[:-1].split('\t')[0]), eval(line[:-1].split('\t')[1])] for line in lines]
    crack_rate = {i: 0 for i in range(9)}
    total = 0
    fin = open(test_file, 'r', encoding="UTF-8")
    while 1:
        line = fin.readline()
        if not line:
            break
        total += 1
        info = line[:-1].split('\t')
        email, pw, username, name, birthday = info[0], info[1], info[2], info[3], info[4].replace('-', '')
        pi = PI(email, pw, name, None, username, None, birthday)
        pi.Convert()
        for index, template in enumerate(templates):
            guess = ""
            for structure, detail in zip(template[0], template[1]):
                if structure[0]!='L' and structure[0]!='D' and structure[0]!='S':
                    if pi.infoMap[structure] == "":
                        guess = None
                        break
                    else:
                        guess += pi.infoMap[structure]
                else:
                    guess += detail
            if guess == pw:
                log = math.ceil(math.log(index+1, 10))
                crack_rate[log] += 1
    s = total
    guessNumber2CrackRateSort = sorted(crack_rate.items(), key=lambda x: x[0])
    guessNumberList = []
    crackRateList = []
    pro = 0
    for i in guessNumber2CrackRateSort:
        guessNumberList.append(i[0])
        pro += i[1] / s
        crackRateList.append(pro)
    for i in crackRateList[:15]:
        print(i, end='\t')
    return {g: c for g, c in zip(guessNumberList[:5], crackRateList[:5])}
def _crack_rate_dataset(dataset):
    trainFile = "data/training_and_testing_PII/{}-train.txt".format(dataset)
    testFile = "data/training_and_testing_PII/{}-test.txt".format(dataset)
    (digitSegmentPro, letterSegmentPro, symbolSegmentPro, templatesPro, digitLengthOrder, letterLengthOrder,
     symbolLengthOrder) = Train(trainFile)
    templateFile = "result/attack/{}-template.txt".format(dataset)
    num = 10000
    MiddleGuess(digitSegmentPro, letterSegmentPro, symbolSegmentPro, templatesPro, digitLengthOrder, letterLengthOrder,
                symbolLengthOrder, num, templateFile)
    return guess(templateFile, testFile)
def crack_rate_datasets():
    dataset2crack_rate = {dataset: _crack_rate_dataset(dataset) for dataset in DATASETS_PII}
    save_pickle(crack_rate_datasets_model.model_path, dataset2crack_rate)
    return
def _crack_rate_category(category):
    datasets = CATEGORY2DATASETS_PII[category]
    crack_rate = {i: 0 for i in range(5)}
    for l in range(5):
        p = [crack_rate_datasets_model.get()[dataset][l] for dataset in datasets]
        q = [DATASET2SIZE[dataset] for dataset in datasets]
        crack_rate[l] = weighted_mean(p, q)
    return crack_rate
def crack_rate_categories():
    category2crack_rate = {category: _crack_rate_category(category) for category in CATEGORIES_PII}
    print(category2crack_rate)
    save_pickle(crack_rate_categories_model.model_path, category2crack_rate)
    return
def to_percent(temp, position):
    return '%1.0f' % (100 * temp) + '%'
def fig():
    x = [1, 10, 100, 1000, 10000]
    financial = [0.018074519483293944, 0.053093131905928265, 0.12262603167350394, 0.14854892748365262, 0.24043096691249718]
    general = [0.007434840316060535, 0.043516341988421706, 0.07401463268926095, 0.0865413876201875, 0.12311115768651383]
    # Markov
    plt.figure(constrained_layout=True, figsize=[8.5, 5])
    font = {'family': 'helvetica', 'size': 17}
    plt.plot(x, financial, label="Financial", color="#6EAE49", linestyle='--', linewidth=3)
    plt.plot(x, general, label="Non-financial", color="#94161F", linestyle='--', linewidth=3)
    # plt.fill_between(x[3:], y_financial[3:], y_general[3:], facecolor = "#D2E3F0",)  # Markov
    plt.legend(fontsize=17,loc='upper left', frameon=False, handlelength=5,labelspacing=1)
    plt.xlabel('Guess Number', font)
    plt.ylabel('Fraction of Cracked Passwords', font)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.xscale('symlog')
    plt.xlim(0.9, 1e4+0.1)
    plt.ylim(-0.002, 0.25) # Markov
    plt.yticks([0,0.05,0.10,0.15,0.2,0.25], fontsize=17)
    plt.xticks([1, 10, 100, 1000, 10000],fontsize=17)
    plt.minorticks_on()
    plt.tick_params(top='in', right='in', which='minor', direction='in')
    plt.tick_params(top='in', right='in', direction='in')
    # plt.savefig(outfile, bbox_inches='tight')
    plt.show()
    return

if __name__ == "__main__":
    # crack_rate_datasets()
    # crack_rate_categories()
    # fig()
    model = crack_rate_datasets_model.get()
    items = list(model['BTC'].keys())
    parameters = {item: {category: [0, 0] for category in CATEGORIES_PII} for item in items}
    for item in items:
        for category in CATEGORIES_PII:
            datasets = CATEGORY2DATASETS_PII[category]
            parameters[item][category][0] += sum(
                [model[dataset][item] * DATASET2SIZE[dataset] for dataset in datasets])
            parameters[item][category][1] = sum([DATASET2SIZE[dataset] for dataset in datasets]) - \
                                            parameters[item][category][0]
        for category in CATEGORIES_PII[1:]:
            observe = np.array([parameters[item][CATEGORIES_PII[0]], parameters[item][category]])
            chi2, p, dof, expected = stats.chi2_contingency(observe)
            print(item, CATEGORIES_PII[0], "vs.", category, chi2, dof, p,
                  parameters[item][CATEGORIES_PII[0]][0] / sum(parameters[item][CATEGORIES_PII[0]]),
                  parameters[item][category][0] / sum(parameters[item][category]))



