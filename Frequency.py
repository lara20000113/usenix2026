def frequency(inFile, outFile):
    pwCount = {}
    fin = open(inFile, 'r', encoding="UTF-8")
    while 1:
        line = fin.readline()
        if not line:
            break
        pw = line[:-1]
        if pw not in pwCount.keys():
            pwCount[pw] = 0
        pwCount[pw] += 1
    fin.close()

    s = sum(pwCount.values())
    fout = open(outFile, 'w', encoding="UTF-8")
    for pw in pwCount.keys():
        fout.write(pw + '\t' + str(pwCount[pw]) + '\t' + str(pwCount[pw]/s) + '\n')
    fout.close()
    return

if __name__ == "__main__":
    frequency("data/QNB.txt", "result/frequency/QNB.txt")
    # fileList = ["CSDN", "Yahoo", "Rockyou", "Dodonew"]
    # for file in fileList:
    #     inFile = "Tmp/" + file + "-Clean.txt"
    #     outFile = "Tmp/" + file + "-frequency.txt"
    #     frequency(inFile, outFile)


