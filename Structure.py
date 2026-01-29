import csv
import operator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats

def structurePCFG(inFile,outFile):
    templateProp = {}
    fin = open(inFile, 'r', encoding="UTF-8")
    while 1:
        line = fin.readline()
        if not line:
            break
        pw = line[:-1].split('\t')[0] + '\n'
        prop = float(line[:-1].split('\t')[2])
        tmpIndex = 0
        dString = ""
        lString = ""
        sString = ""
        templateString = ""
        while pw[tmpIndex] != '\n':
            if pw[tmpIndex].isdigit():
                while pw[tmpIndex].isdigit():
                    dString += pw[tmpIndex]
                    tmpIndex += 1
                templateString = templateString + 'D' + str(len(dString))
                dString = ""
            elif pw[tmpIndex].islower() or pw[tmpIndex].isupper():
                while pw[tmpIndex].islower() or pw[tmpIndex].isupper():
                    lString += pw[tmpIndex]
                    tmpIndex += 1
                templateString = templateString + 'L' + str(len(lString))
                lString = ""
            else:
                while not (pw[tmpIndex].isupper() or pw[tmpIndex].islower() or pw[tmpIndex].isdigit() or pw[
                    tmpIndex] == '\n'):
                    sString += pw[tmpIndex]
                    tmpIndex += 1
                templateString = templateString + 'S' + str(len(sString))
                sString = ""
        if templateString in templateProp.keys():
            templateProp[templateString] += prop
        else:
            templateProp[templateString] = prop
    fin.close()

    fout = open(outFile, 'w')
    for template in templateProp.keys():
        fout.write(template + '\t' + str(templateProp[template]) + '\n')
    fout.close()
    return
def LDS(pw):
    segments = []
    pw += '\n'
    tmpIndex = 0
    dString = ""
    lString = ""
    sString = ""
    templateString = ""
    while pw[tmpIndex] != '\n':
        if pw[tmpIndex].isdigit():
            while pw[tmpIndex].isdigit():
                dString += pw[tmpIndex]
                tmpIndex += 1
            templateString = templateString + 'D'
            segments.append(dString)
            dString = ""
        elif pw[tmpIndex].islower() or pw[tmpIndex].isupper():
            while pw[tmpIndex].islower() or pw[tmpIndex].isupper():
                lString += pw[tmpIndex]
                tmpIndex += 1
            templateString = templateString + 'L'
            segments.append(lString)
            lString = ""
        else:
            while not (pw[tmpIndex].isupper() or pw[tmpIndex].islower() or pw[tmpIndex].isdigit() or
                       pw[tmpIndex] == '\n'):
                sString += pw[tmpIndex]
                tmpIndex += 1
            templateString = templateString + 'S'
            segments.append(sString)
            sString = ""
    return segments, templateString
def structureLDS(inFile, outFile):
    templateProp = {}
    fin = open(inFile, 'r', encoding="UTF-8")
    while 1:
        line = fin.readline()
        if not line:
            break
        pw = line[:-1].split('\t')[0]
        prop = float(line[:-1].split('\t')[2])
        segments, templateString = LDS(pw)
        if templateString in templateProp.keys():
            templateProp[templateString] += prop
        else:
            templateProp[templateString] = prop
    fin.close()

    fout = open(outFile,'w')
    for template in templateProp.keys():
        fout.write(template + '\t' + str(templateProp[template]) + '\n')
    fout.close()
    return
def typeNum(inFile, outFile):
    typeNumProp = {1: 0, 2: 0, 3: 0, 4: 0}
    fin = open(inFile, 'r', encoding="UTF-8")
    while 1:
        line = fin.readline()
        if not line:
            break
        pw = line[:-1].split('\t')[0]
        prop = float(line[:-1].split('\t')[2])
        typeTag = {'L': 0, 'U': 0, 'D': 0, 'S': 0}
        for c in pw:
            if c.islower():
                typeTag['L'] = 1
            elif c.isupper():
                typeTag['U'] = 1
            elif c.isdigit():
                typeTag['D'] = 1
            else:
                typeTag['S'] = 1
        if sum(typeTag.values())>0:
            typeNumProp[sum(typeTag.values())] += prop
    fin.close()

    fout = open(outFile, 'w')
    fout.write('1' + '\t' + str(typeNumProp[1]) + '\n')
    fout.write('2' + '\t' + str(typeNumProp[2]) + '\n')
    fout.write('3' + '\t' + str(typeNumProp[3]) + '\n')
    fout.write('4' + '\t' + str(typeNumProp[4]) + '\n')
    fout.close()
    return
def to_percent(temp, position):
    return '%1.0f' % (100 * float(temp)) + '%'
def drawFig(inFileList, outFile):
    top1 = []
    top2 = []
    top3 = []
    top4 = []
    top5 = []
    for inFile in inFileList:
        structureDic = {}
        fin = open("./result/Structure/" + inFile + "-PCFG.txt", 'r')
        while 1:
            line = fin.readline()
            if not line:
                break
            structure = line[:-1].split('\t')[0]
            prop = float(line[:-1].split('\t')[-1])
            structureDic[structure] = prop
        fin.close()
        structureSorted = sorted(structureDic.items(), key=operator.itemgetter(1), reverse=True)
        top1.append(structureSorted[0][1])
        top2.append(structureSorted[1][1])
        top3.append(structureSorted[2][1])
        top4.append(structureSorted[3][1])
        top5.append(structureSorted[4][1])
    # 柱体底部
    bottom3 = [i + j for i, j in zip(top1, top2)]
    bottom4 = [i + j + k for i, j, k in zip(top1, top2, top3)]
    bottom5 = [i + j + k + l for i, j, k, l in zip(top1, top2, top3, top4)]
    bottom6 = [i + j + k + l + m for i, j, k, l, m in zip(top1, top2, top3, top4, top5)]
    plt.figure(constrained_layout=True, figsize=[17,10]) # 画布大小
    width = 0.9  # 柱子的宽度
    font = {'family': 'helvetica', 'size': 15} # 字体
    # Finaicial
    plt.bar(range(3), top1[:3], width=width, facecolor="#F9A19A", edgecolor='white', linewidth=0.5, alpha=0.9,
            label="Rank-1")
    plt.bar(range(3), top2[:3], width=width, facecolor="#FFCB9D", edgecolor='white', bottom=top1[:3], linewidth=0.5,
             alpha=0.9, label="Rank-2")
    plt.bar(range(3), top3[:3], width=width, facecolor="#FFF59D", edgecolor='white', bottom=bottom3[:3], linewidth=0.5,
             alpha=0.9, label="Rank-3")
    plt.bar(range(3), top4[:3], width=width, facecolor="#C5E1A4", edgecolor='white', bottom=bottom4[:3], linewidth=0.5,
             alpha=0.9, label="Rank-4")
    plt.bar(range(3), top5[:3], width=width, facecolor="#A5D79D", edgecolor='white', bottom=bottom5[:3], linewidth=0.5,
             alpha=0.9, label="Rank-5")
    plt.text(0, top1[0]*0.5, '$\mathregular{L_{6}D_{2}}$', ha='center', va='center', size=15, color='black')
    plt.text(0, top2[0]*0.5+top1[0], '$\mathregular{D_6}$', ha='center', va='center', size=15, color='black')
    plt.text(0, top3[0]*0.5+bottom3[0], '$\mathregular{D_6}$', ha='center', va='center', size=15, color='black')
    plt.text(0, top4[0]*0.5+bottom4[0], '$\mathregular{D_6}$', ha='center', va='center', size=15, color='black')
    plt.text(0, top5[0]*0.5+bottom5[0], '$\mathregular{D_6}$', ha='center', va='center', size=15, color='black')
    plt.text(0, bottom6[0] + 0.01, '22.49%', ha='center', va='center', size=13, color='black')
    plt.text(1, top1[1]*0.5, '$\mathregular{L_6}$', ha='center', va='center', size=15, color='black')
    plt.text(1, top2[1]*0.5+top1[1], '$\mathregular{L_8}$', ha='center', va='center', size=15, color='black')
    plt.text(1, top3[1]*0.5+bottom3[1], '$\mathregular{D_6}$', ha='center', va='center', size=15, color='black')
    plt.text(1, top4[1]*0.5+bottom4[1], '$\mathregular{L_7}$', ha='center', va='center', size=15, color='black')
    plt.text(1, top5[1]*0.5+bottom5[1], '$\mathregular{L_9}$', ha='center', va='center', size=15, color='black')
    plt.text(1, bottom6[1] + 0.01, '34.12%', ha='center', va='center', size=13, color='black')
    plt.text(2, top1[2]*0.5, '$\mathregular{L_{6}D_{2}}$', ha='center', va='center', size=15, color='black')
    plt.text(2, top2[2]*0.5+top1[2], '$\mathregular{L_6}$', ha='center', va='center', size=15, color='black')
    plt.text(2, top3[2]*0.5+bottom3[2], '$\mathregular{L_8}$', ha='center', va='center', size=15, color='black')
    plt.text(2, top4[2]*0.5+bottom4[2], '$\mathregular{L_7}$', ha='center', va='center', size=15, color='black')
    plt.text(2, top5[2]*0.5+bottom5[2], '$\mathregular{L_{4}D_{4}}$', ha='center', va='center', size=15, color='black')
    plt.text(2, bottom6[2]+0.01, '24.09%', ha='center', va='center', size=13, color='black')
    # Social
    plt.bar(range(3, 8), top1[3:8], width=width, facecolor="#F9A19A", edgecolor='white', linewidth=0.5, alpha=0.9)
    plt.bar(range(3, 8), top2[3:8], width=width, facecolor="#FFCB9D", edgecolor='white', bottom=top1[3:8],
             linewidth=0.5, alpha=0.9)
    plt.bar(range(3, 8), top3[3:8], width=width, facecolor="#FFF59D", edgecolor='white', bottom=bottom3[3:8],
             linewidth=0.5, alpha=0.9)
    plt.bar(range(3, 8), top4[3:8], width=width, facecolor="#C5E1A4", edgecolor='white', bottom=bottom4[3:8],
             linewidth=0.5, alpha=0.9)
    plt.bar(range(3, 8), top5[3:8], width=width, facecolor="#A5D79D", edgecolor='white', bottom=bottom5[3:8],
             linewidth=0.5, alpha=0.9)
    plt.text(3, top1[3]*0.5, '$\mathregular{D_{15}}$', ha='center', va='center', size=15, color='black')
    plt.text(3, top2[3]*0.5+top1[3], '$\mathregular{L_6}$', ha='center', va='center', size=15, color='black')
    plt.text(3, top3[3]*0.5+bottom3[3], '$\mathregular{L_8}$', ha='center', va='center', size=15, color='black')
    plt.text(3, top4[3]*0.5+bottom4[3], '$\mathregular{L_7}$', ha='center', va='center', size=15, color='black')
    plt.text(3, top5[3]*0.5+bottom5[3], '$\mathregular{L_6}$', ha='center', va='center', size=15, color='black')
    plt.text(3, bottom6[3] + 0.01, '39.37%', ha='center', va='center', size=13, color='black')
    plt.text(4, top1[4]*0.5, '$\mathregular{D_6}$', ha='center', va='center', size=15, color='black')
    plt.text(4, top2[4]*0.5+top1[4], '$\mathregular{L_6}$', ha='center', va='center', size=15, color='black')
    plt.text(4, top3[4]*0.5+bottom3[4], '$\mathregular{L_8}$', ha='center', va='center', size=15, color='black')
    plt.text(4, top4[4]*0.5+bottom4[4], '$\mathregular{L_7}$', ha='center', va='center', size=15, color='black')
    plt.text(4, top5[4]*0.5+bottom5[4], '$\mathregular{D_8}$', ha='center', va='center', size=15, color='black')
    plt.text(4, bottom6[4] + 0.01, '29.93%', ha='center', va='center', size=13, color='black')
    plt.text(5, top1[5]*0.5, '$\mathregular{L_6}$', ha='center', va='center', size=15, color='black')
    plt.text(5, top2[5]*0.5+top1[5], '$\mathregular{L_8}$', ha='center', va='center', size=15, color='black')
    plt.text(5, top3[5]*0.5+bottom3[5], '$\mathregular{L_{6}D_{2}}$', ha='center', va='center', size=15, color='black')
    plt.text(5, top4[5]*0.5+bottom4[5], '$\mathregular{L_7}$', ha='center', va='center', size=15, color='black')
    plt.text(5, top5[5]*0.5+bottom5[5], '$\mathregular{L_9}$', ha='center', va='center', size=15, color='black')
    plt.text(5, bottom6[5] + 0.01, '25.13%', ha='center', va='center', size=13, color='black')
    plt.text(6, top1[6]*0.5, '$\mathregular{L_8}$', ha='center', va='center', size=15, color='black')
    plt.text(6, top2[6]*0.5+top1[6], '$\mathregular{L_6}$', ha='center', va='center', size=15, color='black')
    plt.text(6, top3[6]*0.5+bottom3[6], '$\mathregular{D_6}$', ha='center', va='center', size=15, color='black')
    plt.text(6, top4[6]*0.5+bottom4[6], '$\mathregular{L_7}$', ha='center', va='center', size=15, color='black')
    plt.text(6, top5[6]*0.5+bottom5[6], '$\mathregular{D_8}$', ha='center', va='center', size=15, color='black')
    plt.text(6, bottom6[6] + 0.01, '77.71%', ha='center', va='center', size=13, color='black')
    plt.text(7, top1[7]*0.5, '$\mathregular{L_{5}D_{5}}$', ha='center', va='center', size=15, color='black')
    plt.text(7, top2[7]*0.5+top1[7], '$\mathregular{L_{4}D_{5}}$', ha='center', va='center', size=15, color='black')
    plt.text(7, top3[7]*0.5+bottom3[7], '$\mathregular{D_6}$', ha='center', va='center', size=15, color='black')
    plt.text(7, top4[7]*0.5+bottom4[7], '$\mathregular{L_{6}D_{2}}$', ha='center', va='center', size=15, color='black')
    plt.text(7, top5[7]*0.5+bottom5[7], '$\mathregular{L_{6}D_{3}}$', ha='center', va='center', size=15, color='black')
    plt.text(7, bottom6[7] + 0.01, '42.15%', ha='center', va='center', size=13, color='black')
    # Email
    plt.bar(range(8, 10), top1[8:10], width=width, facecolor="#F9A19A", edgecolor='white', linewidth=0.5, alpha=0.9)
    plt.bar(range(8, 10), top2[8:10], width=width, facecolor="#FFCB9D", edgecolor='white', bottom=top1[8:10],
            linewidth=0.5, alpha=0.9)
    plt.bar(range(8, 10), top3[8:10], width=width, facecolor="#FFF59D", edgecolor='white', bottom=bottom3[8:10],
            linewidth=0.5, alpha=0.9)
    plt.bar(range(8, 10), top4[8:10], width=width, facecolor="#C5E1A4", edgecolor='white', bottom=bottom4[8:10],
            linewidth=0.5, alpha=0.9)
    plt.bar(range(8, 10), top5[8:10], width=width, facecolor="#A5D79D", edgecolor='white', bottom=bottom5[8:10],
            linewidth=0.5, alpha=0.9)
    plt.text(8, top1[8] * 0.5, '$\mathregular{L_8}$', ha='center', va='center', size=15, color='black')
    plt.text(8, top2[8] * 0.5 + top1[8], '$\mathregular{L_6}$', ha='center', va='center', size=15, color='black')
    plt.text(8, top3[8] * 0.5 + bottom3[8], '$\mathregular{L_7}$', ha='center', va='center', size=15, color='black')
    plt.text(8, top4[8] * 0.5 + bottom4[8], '$\mathregular{D_6}$', ha='center', va='center', size=15, color='black')
    plt.text(8, top5[8] * 0.5 + bottom5[8], '$\mathregular{L_9}$', ha='center', va='center', size=15, color='black')
    plt.text(8, bottom6[8] + 0.01, '35.92%', ha='center', va='center', size=13, color='black')
    plt.text(9, top1[9] * 0.5, '$\mathregular{D_6}$', ha='center', va='center', size=15, color='black')
    plt.text(9, top2[9] * 0.5 + top1[9], '$\mathregular{D_8}$', ha='center', va='center', size=15, color='black')
    plt.text(9, top3[9] * 0.5 + bottom3[9], '$\mathregular{L_8}$', ha='center', va='center', size=15, color='black')
    plt.text(9, top4[9] * 0.5 + bottom4[9], '$\mathregular{L_6}$', ha='center', va='center', size=15, color='black')
    plt.text(9, top5[9] * 0.5 + bottom5[9], '$\mathregular{D_7}$', ha='center', va='center', size=15, color='black')
    plt.text(9, bottom6[9] + 0.01, '43.49%', ha='center', va='center', size=13, color='black')

    # forum
    plt.bar(range(10, 13), top1[10:13], width=width, facecolor="#F9A19A", edgecolor='white', linewidth=0.5, alpha=0.9)
    plt.bar(range(10, 13), top2[10:13], width=width, facecolor="#FFCB9D", edgecolor='white', bottom=top1[10:13],
            linewidth=0.5, alpha=0.9)
    plt.bar(range(10, 13), top3[10:13], width=width, facecolor="#FFF59D", edgecolor='white', bottom=bottom3[10:13],
            linewidth=0.5, alpha=0.9)
    plt.bar(range(10, 13), top4[10:13], width=width, facecolor="#C5E1A4", edgecolor='white', bottom=bottom4[10:13],
            linewidth=0.5, alpha=0.9)
    plt.bar(range(10, 13), top5[10:13], width=width, facecolor="#A5D79D", edgecolor='white', bottom=bottom5[10:13],
            linewidth=0.5, alpha=0.9)
    plt.text(10, top1[10] * 0.5, '$\mathregular{L_6}$', ha='center', va='center', size=15, color='black')
    plt.text(10, top2[10] * 0.5 + top1[10], '$\mathregular{L_8}$', ha='center', va='center', size=15, color='black')
    plt.text(10, top3[10] * 0.5 + bottom3[10], '$\mathregular{L_7}$', ha='center', va='center', size=15, color='black')
    plt.text(10, top4[10] * 0.5 + bottom4[10], '$\mathregular{D_6}$', ha='center', va='center', size=15, color='black')
    plt.text(10, top5[10] * 0.5 + bottom5[10], '$\mathregular{L_9}$', ha='center', va='center', size=15, color='black')
    plt.text(10, bottom6[10] + 0.01, '40.24%', ha='center', va='center', size=13, color='black')
    plt.text(11, top1[11] * 0.5, '$\mathregular{L_6}$', ha='center', va='center', size=15, color='black')
    plt.text(11, top2[11] * 0.5 + top1[11], '$\mathregular{L_8}$', ha='center', va='center', size=15, color='black')
    plt.text(11, top3[11] * 0.5 + bottom3[11], '$\mathregular{L_7}$', ha='center', va='center', size=15, color='black')
    plt.text(11, top4[11] * 0.5 + bottom4[11], '$\mathregular{D_8}$', ha='center', va='center', size=15, color='black')
    plt.text(11, top5[11] * 0.5 + bottom5[11], '$\mathregular{D_6}$', ha='center', va='center', size=15, color='black')
    plt.text(11, bottom6[11] + 0.01, '45.24%', ha='center', va='center', size=13, color='black')
    plt.text(12, top1[12] * 0.5, '$\mathregular{L_6}$', ha='center', va='center', size=15, color='black')
    plt.text(12, top2[12] * 0.5 + top1[12], '$\mathregular{L_7}$', ha='center', va='center', size=15, color='black')
    plt.text(12, top3[12] * 0.5 + bottom3[12], '$\mathregular{L_8}$', ha='center', va='center', size=15, color='black')
    plt.text(12, top4[12] * 0.5 + bottom4[12], '$\mathregular{D_6}$', ha='center', va='center', size=15, color='black')
    plt.text(12, top5[12] * 0.5 + bottom5[12], '$\mathregular{L_9}$', ha='center', va='center', size=15, color='black')
    plt.text(12, bottom6[12] + 0.01, '41.24%', ha='center', va='center', size=13, color='black')

    # Content
    plt.bar(range(13, 17), top1[13:17], width=width, facecolor="#F9A19A", edgecolor='white', linewidth=0.5, alpha=0.9)
    plt.bar(range(13, 17), top2[13:17], width=width, facecolor="#FFCB9D", edgecolor='white', bottom=top1[13:17],
            linewidth=0.5, alpha=0.9)
    plt.bar(range(13, 17), top3[13:17], width=width, facecolor="#FFF59D", edgecolor='white', bottom=bottom3[13:17],
            linewidth=0.5, alpha=0.9)
    plt.bar(range(13, 17), top4[13:17], width=width, facecolor="#C5E1A4", edgecolor='white', bottom=bottom4[13:17],
            linewidth=0.5, alpha=0.9)
    plt.bar(range(13, 17), top5[13:17], width=width, facecolor="#A5D79D", edgecolor='white', bottom=bottom5[13:17],
            linewidth=0.5, alpha=0.9)
    plt.text(13, top1[13] * 0.5, '$\mathregular{L_6}$', ha='center', va='center', size=15, color='black')
    plt.text(13, top2[13] * 0.5 + top1[13], '$\mathregular{L_8}$', ha='center', va='center', size=15, color='black')
    plt.text(13, top3[13] * 0.5 + bottom3[13], '$\mathregular{L_7}$', ha='center', va='center', size=15, color='black')
    plt.text(13, top4[13] * 0.5 + bottom4[13], '$\mathregular{D_4}$', ha='center', va='center', size=15, color='black')
    plt.text(13, top5[13] * 0.5 + bottom5[13], '$\mathregular{D_6}$', ha='center', va='center', size=15, color='black')
    plt.text(13, bottom6[13] + 0.01, '36.05%', ha='center', va='center', size=13, color='black')
    plt.text(14, top1[14] * 0.5, '$\mathregular{L_8}$', ha='center', va='center', size=15, color='black')
    plt.text(14, top2[14] * 0.5 + top1[14], '$\mathregular{L_6}$', ha='center', va='center', size=15, color='black')
    plt.text(14, top3[14] * 0.5 + bottom3[14], '$\mathregular{L_7}$', ha='center', va='center', size=15, color='black')
    plt.text(14, top4[14] * 0.5 + bottom4[14], '$\mathregular{L_{6}D_{2}}$', ha='center', va='center', size=15,
             color='black')
    plt.text(14, top5[14] * 0.5 + bottom5[14], '$\mathregular{L_{7}D_{1}}$', ha='center', va='center', size=15,
             color='black')
    plt.text(14, bottom6[14] + 0.01, '58.73%', ha='center', va='center', size=13, color='black')
    plt.text(15, top1[15] * 0.5, '$\mathregular{L_6}$', ha='center', va='center', size=15, color='black')
    plt.text(15, top2[15] * 0.5 + top1[15], '$\mathregular{L_7}$', ha='center', va='center', size=15, color='black')
    plt.text(15, top3[15] * 0.5 + bottom3[15], '$\mathregular{D_6}$', ha='center', va='center', size=15, color='black')
    plt.text(15, top4[15] * 0.5 + bottom4[15], '$\mathregular{L_8}$', ha='center', va='center', size=15, color='black')
    plt.text(15, top5[15] * 0.5 + bottom5[15], '$\mathregular{L_9}$', ha='center', va='center', size=15, color='black')
    plt.text(15, bottom6[15] + 0.01, '40.76%', ha='center', va='center', size=13, color='black')
    plt.text(16, top1[16] * 0.5, '$\mathregular{L_{6}D_{2}}$', ha='center', va='center', size=15, color='black')
    plt.text(16, top2[16] * 0.5 + top1[16], '$\mathregular{L_6}$', ha='center', va='center', size=15, color='black')
    plt.text(16, top3[16] * 0.5 + bottom3[16], '$\mathregular{L_8}$', ha='center', va='center', size=15, color='black')
    plt.text(16, top4[16] * 0.5 + bottom4[16], '$\mathregular{L_7}$', ha='center', va='center', size=15, color='black')
    plt.text(16, top5[16] * 0.5 + bottom5[16], '$\mathregular{L_{7}D_{2}}$', ha='center', va='center', size=15,
             color='black')
    plt.text(16, bottom6[16] + 0.01, '29.06%', ha='center', va='center', size=13, color='black')

    plt.axvline(2.50, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.axvline(7.50, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.axvline(9.50, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.axvline(12.50, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    # 上边界和右边界去掉
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    # X轴设置
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], inFileList, rotation=45, ha='right',
               rotation_mode='anchor', size=15)
    plt.xlim(-0.5,16.5)
    # Y轴设置
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.10))
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    plt.ylim(0, 0.8)
    plt.yticks(fontsize=15)
    plt.ylabel("Percentage",font)
    # 图例设置
    plt.legend(fontsize=15, labelspacing=1, handletextpad=0.3, frameon=False, bbox_to_anchor=(0.05, 0.8, 0.1, 0.1))
    plt.savefig(outFile, bbox_inches='tight')
    plt.show()
    return
def chi_squared_test(infile):
    lines = open(infile, 'r').readlines()
    for line in lines:
        observe = np.array([[int(line[:-1].split('\t')[1]), int(line[:-1].split('\t')[2])],
                            [int(line[:-1].split('\t')[3]), int(line[:-1].split('\t')[4])]])
        chi2, p, dof, expected = stats.chi2_contingency(observe)
        print(line[:-1].split('\t')[0], chi2, dof, p,
              int(line[:-1].split('\t')[1]) / (int(line[:-1].split('\t')[1])+int(line[:-1].split('\t')[2])),
              int(line[:-1].split('\t')[3]) / (int(line[:-1].split('\t')[3]) + int(line[:-1].split('\t')[4])))




if __name__ == '__main__':
    fileList = ['BTC', 'Clixsense', 'Liveauctioneers', 'Linkedin', 'Twitter', 'Wishbone', 'Badoo', 'Fling',
                'Gmail', 'Hotmail', 'Rootkit', 'Xato', 'Rockyou', 'Yahoo', 'Gawker', 'YouPorn', 'Datpiff']
    # drawFig(fileList, "Structure.pdf") # 画流行口令结构分布图
    chi_squared_test("result/Structure/Data_for_Chi_Squared.txt") # 统计数据卡方检验
    # for file in fileList:
    #     tmp("data/"+file+".txt")









