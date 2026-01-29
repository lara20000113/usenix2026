import operator
import csv
from scipy import stats
import numpy as np

def top(inFile, outFile):
    pwProp = {}
    fin = open(inFile, 'r', encoding="UTF-8")
    while 1:
        line = fin.readline()
        if not line:
            break
        pw = line[:-1].split('\t')[0]
        prop = float(line[:-1].split('\t')[-1])
        pwProp[pw] = prop
    fin.close()

    fout = open(outFile, 'w', encoding="UTF-8")
    pwPropSort = sorted(pwProp.items(), key=operator.itemgetter(1), reverse=True)
    for index in range(10):
        fout.write(pwPropSort[index][0] + '\t' + str(pwPropSort[index][1]) + '\n')
    fout.close()
    return
def drawTable(inFileList, outFile):
    fout = open(outFile, 'w', newline="")
    csvWriter = csv.writer(fout)
    csvWriter.writerow(["DataSets"] + [r for r in range(1, 11)] + ["TotalProportion"])
    for inFile in inFileList:
        print(inFile)
        pws = []
        props = []
        fin = open("./result/TF-IDF/" + inFile + ".txt", 'r')
        c = 0
        while c<10:
            line = fin.readline()
            if not line:
                break
            c += 1
            pw = line[:-1].split('\t')[0]
            prop = float(line[:-1].split('\t')[1])
            pws.append(pw)
            props.append(prop)
        fin.close()
        csvWriter.writerow([inFile] + pws + [sum(props)])
    fout.close()
    return
def popular_password_U_test():
    financial = [1.13, 1.49, 1.04]
    general = [1.17, 1.04, 1.72, 1.06, 2.71, 3.97, 2.05, 2.08, 3.76, 3.93, 1.46, 2.04, 1.58, 4.86, 0.99] # U 9.5 p-value 0.0624
    statistic, p_value = stats.mannwhitneyu(financial, general)
    print(statistic, p_value, sum(financial)/len(financial), sum(general)/len(general))
    return

def chi_squared_test(infile):
    lines = open(infile, 'r').readlines()
    for line in lines:
        observe = np.array([[int(line[:-1].split('\t')[1]), int(line[:-1].split('\t')[2])],
                            [int(line[:-1].split('\t')[3]), int(line[:-1].split('\t')[4])]])
        chi2, p, dof, expected = stats.chi2_contingency(observe)
        print(line[:-1].split('\t')[0], chi2, dof, p,
              int(line[:-1].split('\t')[1]) / (int(line[:-1].split('\t')[1]) + int(line[:-1].split('\t')[2])),
              int(line[:-1].split('\t')[3]) / (int(line[:-1].split('\t')[3]) + int(line[:-1].split('\t')[4]))
              )

if __name__ == '__main__':
    # popular_password_U_test()
    chi_squared_test("result/Popular/Data_for_Chi_Squared.txt")