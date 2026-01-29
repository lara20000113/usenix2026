import numpy as np
from scipy import stats

def type_name(firstname, lastname):
    name0 = firstname + lastname
    name1 = firstname[0] + lastname[0]
    name2 = firstname
    name3 = lastname
    name4 = firstname[0] + lastname
    name5 = firstname + lastname[0]
    return [name0, name1, name2, name3, name4, name5]
def type_birthday(birthday):
    year, month, day = birthday[:4], birthday[4:6], birthday[6:]
    birth0 = birthday
    birth1 = month + day + year
    birth2 = day + month + year
    birth3 = month + day
    birth4 = year
    birth5 = year + month
    birth6 = month + year
    birth7 = year[-2:] + month + year
    birth8 = month + day + year[-2:]
    birth9 = day + month + year[-2:]
    return [birth0, birth1, birth2, birth3, birth4, birth5, birth6, birth7, birth8, birth9]
def LDSParse(string):  # 只分析LDS结构
    string += '\n'
    tmpIndex = 0
    dList = []
    lList = []
    sList = []
    dString = ""
    lString = ""
    sString = ""
    while string[tmpIndex] != '\n':
        if string[tmpIndex].isdigit():
            while string[tmpIndex].isdigit():
                dString += string[tmpIndex]
                tmpIndex += 1
            if len(dString)>=3:
                dList.append(dString)
            dString = ""
        elif string[tmpIndex].islower() or string[tmpIndex].isupper():
            while string[tmpIndex].islower() or string[tmpIndex].isupper():
                lString += string[tmpIndex]
                tmpIndex += 1
            if len(lString) >= 3:
                lList.append(lString)
            lString = ""
        else:
            while not (string[tmpIndex].isupper() or string[tmpIndex].islower() or string[tmpIndex].isdigit() or
                       string[tmpIndex] == '\n'):
                sString += string[tmpIndex]
                tmpIndex += 1
            sList.append(sString)
            sString = ""
    return dList, lList, sList
def type_usename(username):
    dList, lList, sList = LDSParse(username)
    if dList != [] and lList != []:
        return [username, lList[0], dList[0]]
    elif dList != []:
        return [username, dList[0]]
    elif lList != []:
        return [username, lList[0]]
    else:
        return [username]
def type_email(email):
    prefix = email.split('@')[0]
    dList, lList, sList = LDSParse(prefix)
    if dList != [] and lList != []:
        return [email, prefix, lList[0], dList[0]]
    elif dList != []:
        return [email, prefix, dList[0]]
    elif lList != []:
        return [email, prefix, lList[0]]
    else:
        return [email, prefix]
def PII(infile):
    email2infos = {}
    fin = open(infile, 'r', encoding="UTF-8")
    while 1:
        line = fin.readline()
        if not line:
            break
        line = line.lower()
        infos = line[:-1].split('\t')
        if not (len(infos[0])>0 and '@' in infos[1] and infos[1][0]!='@' and len(infos[3])>0 and len(infos[4])>0 and len(infos[5])==10):
            continue
        subtypes_name = type_name(infos[3], infos[4])
        subtypes_birthday = type_birthday(infos[5].replace('-',''))
        subtypes_username = type_usename(infos[0])
        subtypes_email = type_email(infos[1])
        email2infos[infos[1]] = {"name": subtypes_name, "birthday": subtypes_birthday, "username": subtypes_username, "email": subtypes_email}
    fin.close()
    return email2infos
def match(email2infos, infile):
    total, usage_name, usage_birthday, usage_username, usage_email = 0, 0, 0, 0, 0
    fin = open(infile, 'r', encoding="UTF-8")
    while 1:
        try:
            line = fin.readline()
        except:
            continue
        if not line:
            break
        try:
            email = line[:-1].split('\t')[0]
            pw = line[:-1].split('\t')[1].lower()
        except:
            continue
        if email in email2infos.keys():
            total += 1
            for s in email2infos[email]['name']:
                if s in pw:
                    usage_name += 1
                    break
            for s in email2infos[email]['birthday']:
                if s in pw:
                    usage_birthday += 1
                    break
            for s in email2infos[email]['username']:
                if s in pw:
                    usage_username += 1
                    break
            for s in email2infos[email]['email']:
                if s in pw:
                    usage_email += 1
                    break
    fin.close()
    print(total, usage_name, usage_birthday, usage_username, usage_email)
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
    return
def to_percent(temp, position):
    return '%1.0f' % (100 * float(temp)) + '%'
def draw_fig():
    x = np.array([0.106640511, 0.058937619, 0.07658027, 0.059709068])  # 2015年
    y = np.array([0.092029404, 0.049832583, 0.055497826, 0.044144248])  # 2017年
    plt.figure(constrained_layout=True, figsize=[8, 2.5])
    plt.barh([i * 0.24 for i in range(len(y))], -x, color='#F9A19A', label='Financial-related', height=0.2)
    plt.barh([i * 0.24 for i in range(len(x))], y, color='#A5D79D', label='Non-financial', height=0.2)
    plt.text(-0.137, -0.03, "10.66%", fontsize=15)
    plt.text(-0.085, 0.21, "5.89%", fontsize=15)
    plt.text(-0.103, 0.45, "7.66%", fontsize=15)
    plt.text(-0.086, 0.69, "5.97%", fontsize=15)
    plt.text(0.093, -0.03, "9.20%", fontsize=15)
    plt.text(0.05, 0.21, "4.98%", fontsize=15)
    plt.text(0.057, 0.45, "5.55%", fontsize=15)
    plt.text(0.045, 0.69, "4.41%", fontsize=15)
    plt.legend(loc='upper center', ncol=2, fontsize=15, frameon=False, bbox_to_anchor=(0.45, 1.0, 0.1, 0.1))
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.xlim(-0.14, 0.11)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    plt.xticks((-0.1, -0.05, 0, 0.05, 0.1), ('10%', '5%', '0', '5%', '10%'), fontsize=15)
    plt.yticks((0, 0.24, 0.48, 0.72), ('Name', 'Birthday', 'Username', 'Email'), fontsize=15)
    plt.ylim(-0.12, 1.0)
    plt.savefig("PII.pdf", bbox_inches='tight')
    plt.savefig("PII.png", bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    infile = "G:\dataset\ClixSense\initial\ClixSense_PI.csv"
    email2infos = PII(infile)
    fileList = ["Yahoo", "LinkedIn", "DatPiff", "LiveAuctioneers", "ClixSense", "BTC"]
    for file in fileList:
        match(email2infos, "./Data2024/Reuse/{}.txt".format(file))
    chi_squared_test("Result2024/PII/Data_for_Chi_Squared_for_PCFG.txt")
