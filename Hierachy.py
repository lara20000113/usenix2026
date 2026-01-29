from utils import DATASETS
import random
import matplotlib.pyplot as plt
from google.protobuf.internal.wire_format import INT64_MAX

def datav1(metricNumber):
    metric2Result = {i: {file: None for file in DATASETS} for i in range(metricNumber)}
    for inFile in DATASETS:
        fin = open("./result/metrics/" + inFile + '.txt', 'r')
        for i in range(metricNumber):
            line = fin.readline()
            metric2Result[i][inFile] = float(line[:-1])
        fin.close()
    file2Rank = {file: [] for file in DATASETS}
    for metric in metric2Result.keys():
        tmp = sorted(metric2Result[metric].items(), key=lambda x: x[1], reverse=True)
        print('\t'.join([t[0] for t in tmp]))
        for index, i in enumerate(tmp):
            file2Rank[i[0]].append(index)
    return file2Rank
def empirical(file2Rank, metricNumber):
    inFileList = file2Rank.keys()
    ls = []
    for i in range(1,len(inFileList)):
        stable = 0
        first = 0
        second = 0
        abnormal = 0
        score = 0
        level1 = []
        level2 = []
        for file in file2Rank:
            small = len([j for j in file2Rank[file] if j<i])
            big = len([j for j in file2Rank[file] if j>=i])
            if (file2Rank[file][0] < i and small==metricNumber) or (file2Rank[file][0] >= i and big==metricNumber):
                stable += 1
                first = first + 1 if file2Rank[file][0] < i else first
                second = second + 1 if file2Rank[file][0] >= i else second
            else:
                abnormal = abnormal + big if small>big else abnormal + small
            if small >= big:
                level1.append(file)
                score += len(level1)
            else:
                level2.append(file)
                score += len(level2)
        # print(i, stable, first, second, abnormal,level1,level2)
        ls.append([i, stable, first, second, abnormal,level1,level2])
    a = sorted(ls,key=lambda x:x[4])
    for aa in a:
        print(aa)
    print('===========================================================================================================')
    ls = []
    for i1 in range(1,len(inFileList)-1):
        for i2 in range(i1+1,len(inFileList)):
            stable = 0
            first = 0
            second = 0
            third = 0
            abnormal = 0
            level1 = []
            level2 = []
            level3 = []
            for file in file2Rank:
                l1 = len([j for j in file2Rank[file] if j < i1])
                l3 = len([j for j in file2Rank[file] if j >= i2])
                l2 = len([j for j in file2Rank[file] if i1<=j<i2])
                maxL = max([l1,l2,l3])

                if (file2Rank[file][0] < i1 and l1==metricNumber) or \
                    (i1<=file2Rank[file][0]<i2 and l2 == metricNumber) or \
                    (file2Rank[file][0] >= i2 and l3==metricNumber):
                    stable += 1
                    first = first + 1 if file2Rank[file][0] < i1 else first
                    second = second + 1 if file2Rank[file][0] >= i2 else second
                    third = third + 1 if i1<=file2Rank[file][0]<i2 else third
                else:
                    abnormal = abnormal + l2+l3 if maxL==l1 else abnormal + l1+l3 if maxL==l2 else abnormal + l1+l2
                if maxL==l1:
                    level1.append(file)
                elif maxL==l2:
                    level2.append(file)
                else:
                    level3.append(file)
            # print(i1,i2, stable, first, second, third, abnormal, level1, level2, level3)
            ls.append([i1,i2, stable, first, second, third, abnormal, level1, level2, level3])
    a = sorted(ls,key=lambda x:x[6])
    for aa in a:
        if len(aa[7])>1 and len(aa[8])>1 and len(aa[9])>1:
            print(aa)
    print('===========================================================================================================')
    for i1 in range(1,len(inFileList)-2):
        for i2 in range(i1+1,len(inFileList)-1):
            for i3 in range(i2 + 1, len(inFileList)):
                stable = 0
                first = 0
                second = 0
                third = 0
                fourth = 0
                abnormal = 0
                level1 = []
                level2 = []
                level3 = []
                level4 = []
                for file in file2Rank:
                    l1 = len([j for j in file2Rank[file] if j < i1])
                    l3 = len([j for j in file2Rank[file] if i3>j >= i2])
                    l2 = len([j for j in file2Rank[file] if i1<=j<i2])
                    l4 = len([j for j in file2Rank[file] if j >= i3])
                    maxL = max([l1,l2,l3,l4])
                    if (file2Rank[file][0] < i1 and l1==metricNumber) or \
                        (i1<=file2Rank[file][0]<i2 and l2 == metricNumber) or \
                        (i3>file2Rank[file][0] >= i2 and l3==metricNumber) or \
                        (file2Rank[file][0] >= i3 and l4 == metricNumber)   :
                        stable += 1
                        first = first + 1 if file2Rank[file][0] < i1 else first
                        second = second + 1 if i3>file2Rank[file][0] >= i2 else second
                        third = third + 1 if i1<=file2Rank[file][0]<i2 else third
                        fourth = fourth + 1 if file2Rank[file][0] >= i3 else fourth
                    else:
                        abnormal = abnormal + l2+l3+l4 if maxL==l1 else abnormal + l1+l3+l4 if maxL==l2 else abnormal + l1+l2+l4 if maxL==l3 else abnormal + l1+l2+l3
                    if maxL==l1:
                        level1.append(file)
                    elif maxL==l2:
                        level2.append(file)
                    elif maxL==l3:
                        level3.append(file)
                    else:
                        level4.append(file)
                print(i1,i2,i3, stable, first, second, third,fourth, abnormal, level1, level2, level3,level4)
def loss(group, exception=None, addition=None):
    _group = {key: group[key] for key in group}
    if addition:
        _group[addition[0]] = addition[1:]
    if exception:
        del _group[exception]
    return sum([item[0] for item in _group.values()]) * sum([item[1] for item in _group.values()])
def move():
    L = [157, 140, 160, 94, 101, 99, 98, 101, 103, 109, 40, 50, 43, 56, 52, 48, 56]
    P = [0.25270751, 0.282613155, 0.227256282, 0.248577608, 0.341978548, 0.298613301, 0.146709201, 0.538130672,
         0.344974367, 0.451137599, 0.328858294, 0.442520651, 0.455373062, 0.394312645, 0.428196323, 0.499834988,
         0.385158212]
    L_P = [l / p for l, p in zip(L, P)]
    accounts = [(dataset, l, p, l_p) for dataset, l, p, l_p in zip(DATASETS, L, P, L_P)]
    accounts_sort = sorted(accounts, key=lambda x: x[3])
    group1, group2, group3 = ({item[0]: [item[1], item[2]] for item in accounts_sort[:5]},
                              {item[0]: [item[1], item[2]] for item in accounts_sort[5:12]},
                              {item[0]: [item[1], item[2]] for item in accounts_sort[12:]})
    move = True
    iteration = 0
    while move:
        iteration += 1
        original_loss = loss(group1) + loss(group2) + loss(group3)
        move = False
        for dataset in group1.keys():
            if loss(group1, exception=dataset) + loss(group2, addition=[dataset, group1[dataset][0], group1[dataset][1]]) + loss(group3) < original_loss:
                group2[dataset] = group1[dataset][:]
                del group1[dataset]
                move = True
                break
            if loss(group1, exception=dataset) + loss(group3, addition=[dataset, group1[dataset][0], group1[dataset][1]]) + loss(group2) < original_loss:
                group3[dataset] = group1[dataset][:]
                del group1[dataset]
                move = True
                break
        if move:
            continue
        for dataset in group2.keys():
            if loss(group2, exception=dataset) + loss(group1, addition=[dataset, group2[dataset][0], group2[dataset][1]]) + loss(group3) < original_loss:
                group1[dataset] = group2[dataset][:]
                del group2[dataset]
                move = True
                break
            if loss(group2, exception=dataset) + loss(group3, addition=[dataset, group2[dataset][0], group2[dataset][1]]) + loss(group1) < original_loss:
                group3[dataset] = group2[dataset][:]
                del group2[dataset]
                move = True
                break
        if move:
            continue
        for dataset in group3.keys():
            if loss(group3, exception=dataset) + loss(group1, addition=[dataset, group3[dataset][0], group3[dataset][1]]) + loss(group2) < original_loss:
                group1[dataset] = group3[dataset][:]
                del group3[dataset]
                move = True
                break
            if loss(group3, exception=dataset) + loss(group2, addition=[dataset, group3[dataset][0], group3[dataset][1]]) + loss(group1) < original_loss:
                group2[dataset] = group3[dataset][:]
                del group3[dataset]
                move = True
                break
    L1, P1 = sum([item[0] for item in group1.values()]), sum([item[1] for item in group1.values()])
    L2, P2 = sum([item[0] for item in group2.values()]), sum([item[1] for item in group2.values()])
    L3, P3 = sum([item[0] for item in group3.values()]), sum([item[1] for item in group3.values()])
    return group1, group2, group3, L1/P1, L2/P2, L3/P3
def fig():
    plt.figure(constrained_layout=True, figsize=[8.5, 5])
    group1, group2, group3, L1_P1, L2_P2, L3_P3 = move()
    print(group1)
    print(group2)
    print(group3)
    print(L1_P1, L2_P2, L3_P3)
    dataset1 = [dataset for dataset in group1.keys()]
    dataset2 = [dataset for dataset in group2.keys()]
    dataset3 = [dataset for dataset in group3.keys()]
    plt.scatter([group1[dataset][1] for dataset in dataset1], [group1[dataset][0] for dataset in dataset1],
                edgecolor='#6EAE49', facecolor='white', marker='o', s=200, linewidths=2, label="Group 1")
    plt.scatter([group2[dataset][1] for dataset in dataset2], [group2[dataset][0] for dataset in dataset2],
               edgecolor='#4F5FA8', facecolor='white', marker='o', s=200, linewidths=2, label="Group 2")
    plt.scatter([group3[dataset][1] for dataset in dataset3], [group3[dataset][0] for dataset in dataset3],
                edgecolor='#EA5029', facecolor='white', marker='o', s=200, linewidths=2, label="Group 3")
    plt.text(group1[dataset1[0]][1]+0.01, group1[dataset1[0]][0]-2, dataset1[0], size=13)
    plt.text(group1[dataset1[1]][1]-0.02, group1[dataset1[1]][0]+5, dataset1[1], size=13)
    plt.text(group1[dataset1[2]][1] - 0.025, group1[dataset1[2]][0] + 6, dataset1[2], size=13)
    plt.text(group1[dataset1[3]][1]-0.02, group1[dataset1[3]][0]-10, dataset1[3], size=13)
    plt.text(group1[dataset1[4]][1] - 0.015, group1[dataset1[4]][0]+5, dataset1[4], size=13)
    plt.text(group1[dataset1[5]][1] - 0.02, group1[dataset1[5]][0] + 5, dataset1[5], size=13)
    plt.text(group1[dataset1[6]][1], group1[dataset1[6]][0]-8, dataset1[6], size=13)
    plt.text(group2[dataset2[0]][1]-0.01, group2[dataset2[0]][0]+4, dataset2[0], size=13)
    plt.text(group2[dataset2[1]][1] - 0.02, group2[dataset2[1]][0] -8, dataset2[1], size=13)
    plt.text(group2[dataset2[2]][1] - 0.01, group2[dataset2[2]][0] + 7, dataset2[2], size=13)
    plt.text(group2[dataset2[3]][1] - 0.02, group2[dataset2[3]][0]-10, dataset2[3], size=13)
    plt.text(group2[dataset2[4]][1] - 0.025, group2[dataset2[4]][0]+4, dataset2[4], size=13)
    plt.text(group3[dataset3[0]][1]-0.02, group3[dataset3[0]][0]-8, dataset3[0], size=13)
    plt.text(group3[dataset3[1]][1] - 0.03, group3[dataset3[1]][0] - 8, dataset3[1], size=13)
    plt.text(group3[dataset3[2]][1]-0.01, group3[dataset3[2]][0]-8, dataset3[2], size=13)
    plt.text(group3[dataset3[3]][1] - 0.01, group3[dataset3[3]][0]+5, dataset3[3], size=13)
    plt.text(group3[dataset3[4]][1] - 0.07, group3[dataset3[4]][0]-8, dataset3[4], size=13)
    plt.plot([0, 1], [0, L1_P1], color='#6EAE49')
    plt.plot([0, 1], [0, L2_P2], color='#4F5FA8')
    plt.plot([0, 1], [0, L3_P3], color='#EA5029')
    plt.legend(fontsize=15, frameon=False, bbox_to_anchor=(0.85, 0.7))
    plt.ylim(35, 165)
    plt.xlim(0.13, 0.55)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.xlabel('Probability (P)', fontsize=17)
    plt.ylabel('Loss (L)', fontsize=17)
    plt.minorticks_on()
    plt.show()
    return




if __name__ == '__main__':
    # data = datav1(15) # metrics
    # empirical(data, 15)
    fig()
