import random

def valid_pw(pw):
    if len(pw) > 30:
        return False
    for c in pw:
        if ord(c) <= 32 or ord(c) > 126:
            return False
    return True
def cleanPWs(inFile, outFile):
    fin = open(inFile, 'r', encoding="UTF-8")
    fout = open(outFile, 'w', encoding="UTF-8")
    while 1:
        line = fin.readline()
        if not line:
            break
        pw = line[:-1]
        if not valid_pw(pw):
            continue
        fout.write(pw + '\n')
    fin.close()
    fout.close()
    return
def cleanEmailPW(infile, outfile, flag):
    fin = open(infile, 'r', encoding="UTF-8")
    fout = open(outfile, 'w', encoding="UTF-8")
    while 1:
        try:
            line = fin.readline()
        except UnicodeDecodeError:
            continue
        if not line:
            break
        # print(line[:-1])
        try:
            email = line[:-1].split('\t')[0]
            pw = line[:-1].split('\t')[1]
            if '@' not in email or email[0] == '@' or not valid_pw(pw):
                continue
            fout.write(email + '\t' + pw + '\t' + flag + '\n')
        except IndexError:
            continue
    fin.close()
    fout.close()
    return
def split(infile, outfile1, outfile2, pro): # 划分训练集和测试集 5：5
    fin = open(infile, 'r', encoding="UTF-8")
    fout1 = open(outfile1, 'w', encoding="UTF-8")
    fout2 = open(outfile2, 'w', encoding="UTF-8")
    while 1:
        line = fin.readline()
        if not line:
            break
        if random.random() < pro:
            fout2.write(line)
        else:
            fout1.write(line)
    fin.close()
    fout1.close()
    fout2.close()
    return
def add(inFile, outFile):
    fin = open(inFile, 'r', encoding="UTF-8")
    fout = open(outFile, 'w', encoding="UTF-8")
    while 1:
        line = fin.readline()
        if not line:
            break
        pairs = eval(line[:-1])
        pw1 = pairs[0]
        pw2 = pairs[1]
        if (pw1.isdigit() and len(pw1) == 30) or (pw2.isdigit() and len(pw2) == 30):
            continue
        # fout.write('*'+'\t'+pw1+'\t'+pw2+'\t'+'*'+'\n') #train
        fout.write('*'+'\t'+pw1+'\t'+pw2+'\n') # test
    fin.close()
    fout.close()
    return



if __name__ == "__main__":
    fileList = ['BTC', 'ClixSense', 'LiveAuctioneers', 'LinkedIn', 'Twitter', 'Wishbone', 'Badoo', 'Fling',
                'Mate1', 'Rockyou', 'Gmail', 'Hotmail', 'Rootkit', 'Xato', 'Yahoo', 'Gawker', 'YouPorn', 'DatPiff']
    # for file in fileList:
    #     split(file) # 划分漫步攻击实验的训练集和测试集
    # fileList = ['Fin-Fin', "Non-Non", "Fin-Non", "Non-Fin"]
    # pros = [0.0060589169080135, 0.0064195153265928, 7.122882545091408e-4, 7.122882545091408e-4]
    # for file, pro in zip(fileList, pros):
    #     split(file+".txt", file+"-train.txt", file+"-test.txt", pro) # targuessII划分训练集和测试集
    # for file in fileList:
    #     split("Result2024/semantic/"+file+".txt", "Data2024/semantic/"+file+"-train.txt",
    #           "Data2024/semantic/"+file+"-test.txt", 0.5)
    # cleanPWs("I:\dataset\QNB\QNB_PWonly(92212).txt", "data/QNB.txt")
    cleanEmailPW("G:\\dataset\\rootkit.txt", "./Data2024/Reuse/rootkit.txt", "Rootkit")






