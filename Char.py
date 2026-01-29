import csv

def type(inFile,outFile):
    typeDic = {'L': 0, 'U': 0, 'D': 0, 'S': 0}
    fin = open(inFile, 'r', encoding="UTF-8")
    while 1:
        line = fin.readline()
        if not line:
            break
        pw = line[:-1].split('\t')[0]
        count = float(line[:-1].split('\t')[1])
        for c in pw:
            if c.isdigit():
                typeDic['D'] += count
            elif c.islower():
                typeDic['L'] += count
            elif c.isupper():
                typeDic['U'] += count
            else:
                typeDic['S'] += count
    fin.close()

    fout = open(outFile, 'w')
    fout.write("L" + '\t' + str(typeDic['L']/sum(typeDic.values())) + '\n')
    fout.write("D" + '\t' + str(typeDic['D']/sum(typeDic.values())) + '\n')
    fout.write("U" + '\t' + str(typeDic['U']/sum(typeDic.values())) + '\n')
    fout.write("S" + '\t' + str(typeDic['S']/sum(typeDic.values())) + '\n')
    fout.close()
    return

def drawTableType(inFileList, outFile):
    fout = open(outFile, 'w', newline="")
    csvWriter = csv.writer(fout)
    csvWriter.writerow(["DataSets"] + ['L', 'D', 'U', 'S'])
    for inFile in inFileList:
        fin = open("./result/char/" + inFile + "-Type.txt", 'r')
        line = fin.readline()
        propL = float(line[:-1].split('\t')[1])
        line = fin.readline()
        propD = float(line[:-1].split('\t')[1])
        line = fin.readline()
        propU = float(line[:-1].split('\t')[1])
        line = fin.readline()
        propS = float(line[:-1].split('\t')[1])
        csvWriter.writerow([inFile, propL, propD, propU, propS])
        fin.close()
    fout.close()
    return

def upperAndSpecial(inFile, outFile):
    typeProp = {'U': 0, 'S': 0, 'US': 0, 'Other': 0}
    fin = open(inFile, 'r', encoding="UTF-8")
    while 1:
        line = fin.readline()
        if not line:
            break
        pw = line[:-1].split('\t')[0]
        prop = float(line[:-1].split('\t')[2])
        upper = False
        special = False
        for c in pw:
            if c.isupper():
                upper = True
            elif c.islower() or c.isdigit():
                pass
            else:
                special = True
        if upper==True and special==True:
            typeProp['US'] += prop
        elif upper == True:
            typeProp['U'] += prop
        elif special == True:
            typeProp['S'] += prop
        else:
            typeProp['Other'] += prop
    fin.close()

    fout = open(outFile, 'w')
    fout.write('Other' + '\t' + str(typeProp['Other']) + '\n')
    fout.write('U' + '\t' + str(typeProp['U']) + '\n')
    fout.write('S' + '\t' + str(typeProp['S']) + '\n')
    fout.write('US' + '\t' + str(typeProp['US']) + '\n')
    fout.close()
    return

def drawTableUS(inFileList, outFile):
    fout = open(outFile, 'w', newline="")
    csvWriter = csv.writer(fout)
    csvWriter.writerow(["DataSets"] + ['Other', 'U', 'S', 'US'])
    for inFile in inFileList:
        fin = open("./result/char/" + inFile + "-US.txt", 'r')
        line = fin.readline()
        propOther = float(line[:-1].split('\t')[1])
        line = fin.readline()
        propU = float(line[:-1].split('\t')[1])
        line = fin.readline()
        propS = float(line[:-1].split('\t')[1])
        line = fin.readline()
        propUS = float(line[:-1].split('\t')[1])
        csvWriter.writerow([inFile, propOther, propU, propS, propUS])
        fin.close()
    fout.close()
    return

def aveTypeNum(inFile, outFile):
    typeDic = {'L': 0, 'U': 0, 'D': 0, 'S': 0}
    s = 0
    fin = open(inFile, 'r', encoding="UTF-8")
    while 1:
        line = fin.readline()
        if not line:
            break
        pw = line[:-1].split('\t')[0]
        count = float(line[:-1].split('\t')[1])
        s += count
        for c in pw:
            if c.isdigit():
                typeDic['D'] += count
            elif c.islower():
                typeDic['L'] += count
            elif c.isupper():
                typeDic['U'] += count
            else:
                typeDic['S'] += count
    fin.close()
    fout = open(outFile, 'w')
    fout.write(str(typeDic['L']/s)+'\n')
    fout.write(str(typeDic['D']/s)+'\n')
    fout.write(str(typeDic['U']/s)+'\n')
    fout.write(str(typeDic['S']/s)+'\n')
    fout.close()
    return

def drawTableAdvType(inFileList, outFile):
    fout = open(outFile, 'w', newline="")
    csvWriter = csv.writer(fout)
    csvWriter.writerow(["DataSets"] + ['L', 'D', 'U', 'S'])
    for inFile in inFileList:
        fin = open("./result/char/" + inFile + "-advType.txt", 'r')
        line = fin.readline()
        LProp = float(line[:-1])
        line = fin.readline()
        DProp = float(line[:-1])
        line = fin.readline()
        UProp = float(line[:-1])
        line = fin.readline()
        SProp = float(line[:-1])
        csvWriter.writerow([inFile, LProp, DProp, UProp, SProp])
        fin.close()
    fout.close()
    return

def letterArrangement(inFileList, outFile):
    # fout = open(outFile, 'w')
    char2Frequency = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0, 'f': 0, 'g': 0, 'h': 0, 'i': 0, 'j': 0, 'k': 0, 'l': 0,
                      'm': 0, 'n': 0, 'o': 0, 'p': 0, 'q': 0, 'r': 0, 's': 0, 't': 0, 'u': 0, 'v': 0, 'w': 0, 'x': 0,
                      'y': 0, 'z': 0}
    for inFile in inFileList:
        # char2Frequency = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0, 'f': 0, 'g': 0, 'h': 0, 'i': 0, 'j': 0, 'k': 0, 'l': 0,
        #                   'm': 0, 'n': 0, 'o': 0, 'p': 0, 'q': 0, 'r': 0, 's': 0, 't': 0, 'u': 0, 'v': 0, 'w': 0, 'x': 0,
        #                   'y': 0, 'z': 0}
        fin = open('./result/frequency/'+inFile+'.txt', 'r')
        while 1:
            line = fin.readline()
            if not line:
                break
            pw = line[:-1].split('\t')[0]
            count = int(line[:-1].split('\t')[1])
            for c in pw:
                if c.islower():
                    char2Frequency[c] += count
        fin.close()
    char2FrequencySort = sorted(char2Frequency.items(), key=lambda x: x[1], reverse=True)
    print(''.join([i[0] for i in char2FrequencySort]))
    # fout.write('AllFinancial' + '\t' + ''.join([i[0] for i in char2FrequencySort]) + '\n')
    # fout.close()
    return

def reversionNumber(inFile, outFile):
    fin = open(inFile, 'r')
    type2String = {line[:-1].split('\t')[0]: line[:-1].split('\t')[1] for line in fin.readlines()}
    fin.close()
    allType = type2String.keys()
    rDic = {key1: {key2: 0 for key2 in allType} for key1 in allType}
    for key1 in allType:
        for key2 in allType:
            s1 = type2String[key1]
            s2 = type2String[key2]
            char2Number = {char: index for index, char in enumerate(s1)}
            s2Convert = [char2Number[char] for char in s2]
            r = 0
            for index1, n1 in enumerate(s2Convert):
                for index2, n2 in enumerate(s2Convert[:index1]):
                    r = r + 1 if n2 > n1 else r
            rDic[key1][key2] = r
    fout = open(outFile, 'w', newline="")
    csvWriter = csv.writer(fout)
    for key in allType:
        csvWriter.writerow([key]+[rDic[key][key1] for key1 in allType])
    fout.close()
    return

if __name__ == "__main__":
    fileList = ['BTC', 'Clixsense', 'NeoPets', 'Liveauctioneers', 'Linkedin', 'Twitter', 'Wishbone', 'Badoo', 'Fling',
                'Mate1', 'Rockyou', 'Gmail', 'Hotmail', 'Rootkit', 'Xato', 'Yahoo', 'Gawker', 'YouPorn', 'Datpiff']
    # for file in fileList:
    #     inFile = "./result/frequency/" + file + ".txt"
    #     outFile = "./result/char/" + file + "-Type.txt"
    #     type(inFile, outFile)
    #     outFile = "./result/char/" + file + "-US.txt"
    #     upperAndSpecial(inFile, outFile)
    #     outFile = "./result/char/" + file + "-advType.txt"
    #     aveTypeNum(inFile, outFile)
    # drawTableType(fileList, "char-Type.csv")
    # drawTableUS(fileList, 'char-US.csv')
    # drawTableAdvType(fileList, "char-AdvType.csv")
    # letterArrangement(fileList, 'LetterArrangement.txt')
    # reversionNumber('LetterArrangement.txt', 'ReversionNumber.csv')
    type("./result/frequency/QNB.txt", "./result/char/QNB-Type.txt")



