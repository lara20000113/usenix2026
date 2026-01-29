from itertools import combinations, product
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

def parser(inFile):
    fin = open(inFile, 'r', encoding="UTF-8")
    while 1:
        line = fin.readline()
        if not line:
            break
        email = line[:-1].split('\t')[0]
        pw = line[:-1].split('\t')[1]
        website = line[:-1].split('\t')[2]
        if website not in ["BTC", "ClixSense", "LiveAuctioneers", "LinkedIn", "Yahoo", "DatPiff", "Twitter", "Rootkit"] or '@' not in email:
            continue
        yield email, pw, email.split('@')[0], website, True
    fin.close()
    yield None, None, None, None, False
def root(nodes, index):
    if nodes[index] == index:
        return index
    return root(nodes, nodes[index])
def union_find_set(pwlist_based_email):
    emails = list(pwlist_based_email.keys())
    n = len(emails)
    nodes = [i for i in range(n)]
    for index1, email1 in enumerate(emails):
        for index2, email2 in enumerate(emails):
            if index1 >= index2:
                continue
            if len(set(pwlist_based_email[email1])&set(pwlist_based_email[email2]))>0:
                if nodes[index1] > nodes[index2]:
                    nodes[index1] = nodes[index2]
                elif nodes[index1] < nodes[index2]:
                    nodes[index2] = nodes[index1]
    roots = [root(nodes, index) for index in range(n)] # 每个元素的根节点
    clusters = {r: [] for r in set(roots)}
    for index, r in enumerate(roots):
        clusters[r].append(index)
    results = [[emails[i] for i in clusters[r]] for r in clusters.keys()]
    return results
def union(inFile, outFile):
    g = parser(inFile)
    fout = open(outFile, 'w', encoding="UTF-8")
    email, pw, prefix, website, status = next(g)
    prefix0 = prefix
    while status:
        pwlist_based_email = {} # 邮箱对应的口令列表，所有邮箱前缀相同
        pwlist_based_email_include_website = {}  # 邮箱对应的口令列表，所有邮箱前缀相同，包括网站
        while prefix == prefix0:
            if email not in pwlist_based_email.keys():
                pwlist_based_email[email] = []
                pwlist_based_email_include_website[email] = []
            pwlist_based_email[email].append(pw)
            pwlist_based_email_include_website[email].append([pw, website])
            email, pw, prefix, website, status = next(g)
        prefix0 = prefix
        assert len(pwlist_based_email.keys()) > 0
        joins = union_find_set(pwlist_based_email)
        for join in joins: # 每个邮箱列表
            pwlist_based_website = {}
            for j in join: # 每个邮箱
                for pair in pwlist_based_email_include_website[j]:
                    pwlist_based_website[pair[1]] = pair[0]
            pwlist_financial = []
            pwlist_general = []
            for web in pwlist_based_website.keys():
                if web in ["BTC", "ClixSense", "LiveAuctioneers"]:
                    pwlist_financial.append(pwlist_based_website[web])
                else:
                    assert web in ["LinkedIn", "Yahoo", "DatPiff", "Twitter", "Rootkit"]
                    pwlist_general.append(pwlist_based_website[web])
            fout.write(str(join)+'\t'+str(pwlist_financial)+'\t'+str(pwlist_general)+'\n')
    return
def pairs(inFile, outFile1, outFile2, outFile3, outFile4):
    fin = open(inFile, 'r', encoding="UTF-8")
    fout1 = open(outFile1, 'w')
    fout2 = open(outFile2, 'w')
    fout3 = open(outFile3, 'w')
    fout4 = open(outFile4, 'w')
    finfin = 0
    nonnon = 0
    finnon = 0
    nonfin = 0
    while 1:
        line = fin.readline()
        if not line:
            break
        finPWs = eval(line[:-1].split('\t')[1])
        nonPWs = eval(line[:-1].split('\t')[2])
        finPWsClean = [pw for pw in finPWs]
        nonPWsClean = [pw for pw in nonPWs]
        assert len(finPWsClean) + len(nonPWsClean) <= 8
        pairFinFin = list(combinations(finPWsClean, 2))
        for p in pairFinFin:
            fout1.write(str(p)+'\n')
            finfin += 1
        pairNonNon = list(combinations(nonPWsClean, 2))
        for p in pairNonNon:
            fout2.write(str(p) + '\n')
            nonnon += 1
        pairFinNon = product(finPWsClean, nonPWsClean)
        for p in pairFinNon:
            fout4.write(str(p)+'\n')
            finnon += 1
        pairNonFin = product(nonPWsClean, finPWsClean)
        for p in pairNonFin:
            fout3.write(str(p) + '\n')
            nonfin += 1
    fin.close()
    fout1.close()
    fout2.close()
    fout3.close()
    fout4.close()
    # 165046 155775 1403926 1403926 1568972 1559701
    print(finfin, nonnon, nonfin, finnon)
    return
def LD(str1, str2):
    # 计算编辑距离
    l1 = len(str1)
    l2 = len(str2)
    dp = [[0 for j in range(l2+1)] for i in range(l1+1)]
    for i in range(l2+1):
        dp[0][i] = i
    for i in range(l1+1):
        dp[i][0] = i
    for i in range(1, l1+1):
        for j in range(1, l2+1):
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(str1[i-1]!=str2[j-1]))
    # 路径回溯
    pos1 = l1
    pos2 = l2
    td_str = ""
    ti_str = ""
    hd_str = ""
    hi_str = ""
    replace_pair = []
    while pos1+pos2>0:
        # 水塘抽样决定哪一条路径
        sample = 0
        if pos1 > 0 and dp[pos1][pos2] == dp[pos1-1][pos2] + 1:
            sample += 1
            if random.random()<(1/sample):
                direction = "Delete"
        if pos2 > 0 and dp[pos1][pos2] == dp[pos1][pos2-1] + 1:
            sample += 1
            if random.random() < (1 / sample):
                direction = "Insert"
        if pos1 > 0 and pos2 > 0 and dp[pos1][pos2] == dp[pos1-1][pos2-1] + (str1[pos1-1]!=str2[pos2-1]):
            sample += 1
            if random.random() < (1 / sample):
                direction = "Replace"
                if str1[pos1-1] == str2[pos2-1]:
                    replace_pair.append([pos1-1, pos2-1])
        assert sample > 0
        # 计算首尾部插入和删除操作
        if direction == 'Delete' and pos2 == l2:
            td_str = str1[pos1-1] + td_str
        elif direction == 'Insert' and pos1 == l1:
            ti_str = str2[pos2-1] + ti_str
        elif direction == 'Delete' and pos2 == 0:
            hd_str = str1[pos1 - 1] + hd_str
        elif direction == 'Insert' and pos1 == 0:
            hi_str = str2[pos2 - 1] + hi_str
        if direction == 'Delete':
            pos1 -= 1
        elif direction == 'Insert':
            pos2 -= 1
        else:
            assert direction == 'Replace'
            pos1 -= 1
            pos2 -= 1
    return dp[l1][l2]/max(l1,l2), dp[l1][l2], td_str, ti_str, hd_str, hi_str, replace_pair
def capitalize(pw): # 大小写操作
    pw_list = []
    type_list = []
    if pw[0].islower(): # 首字母大写
        pw_list.append(pw[0].upper()+pw[1:])
        type_list.append(0)
    if pw[0].isupper(): # 首字母小写
        pw_list.append(pw[0].lower()+pw[1:])
        type_list.append(1)
    if pw.lower() not in pw_list: # 全小写
        pw_list.append(pw.lower())
        type_list.append(2)
    if pw.upper() not in pw_list: # 全大写
        pw_list.append(pw.upper())
        type_list.append(3)
    return pw_list, type_list
def leet(pw): # 跳变
    pw_list = []
    type_list = []
    if 'a' in pw:
        pw_list.append(pw.replace('a','@'))
        type_list.append(0)
    if '@' in pw:
        pw_list.append(pw.replace('@','a'))
        type_list.append(1)
    if 's' in pw:
        pw_list.append(pw.replace('s','$'))
        type_list.append(2)
    if '$' in pw:
        pw_list.append(pw.replace('$','s'))
        type_list.append(3)
    if 'o' in pw:
        pw_list.append(pw.replace('o','0'))
        type_list.append(4)
    if '0' in pw:
        pw_list.append(pw.replace('0','o'))
        type_list.append(5)
    if 'i' in pw:
        pw_list.append(pw.replace('i','1'))
        type_list.append(6)
    if '1' in pw:
        pw_list.append(pw.replace('1','i'))
        type_list.append(7)
    if 'e' in pw:
        pw_list.append(pw.replace('e','3'))
        type_list.append(8)
    if '3' in pw:
        pw_list.append(pw.replace('3','e'))
        type_list.append(9)
    return pw_list, type_list
def segment(pw):  # 分段
    pw += '\n'
    tmpIndex = 0
    seg = ""
    seg_list = []
    template = ""
    while pw[tmpIndex] != '\n':
        if pw[tmpIndex].isdigit():
            while pw[tmpIndex].isdigit():
                seg += pw[tmpIndex]
                tmpIndex += 1
            seg_list.append(seg)
            seg = ""
            template += 'D'
        elif pw[tmpIndex].islower() or pw[tmpIndex].isupper():
            while pw[tmpIndex].islower() or pw[tmpIndex].isupper():
                seg += pw[tmpIndex]
                tmpIndex += 1
            seg_list.append(seg)
            seg = ""
            template += 'L'
        else:
            while not (pw[tmpIndex].isupper() or pw[tmpIndex].islower() or pw[tmpIndex].isdigit() or
                       pw[tmpIndex] == '\n'):
                seg += pw[tmpIndex]
                tmpIndex += 1
            seg_list.append(seg)
            seg = ""
            template += 'S'
    return seg_list, template
def reverse(pw): # 反转
    pw_list = []
    type_list = []
    pw_list.append(pw[::-1])
    type_list.append(0)
    seg_list = segment(pw)[0]
    type2 = ''.join([seg[::-1] for seg in seg_list])
    if type2 not in pw_list:
        pw_list.append(type2)
        type_list.append(1)
    return pw_list, type_list
def SM(pw):
    pw_list = []
    type_list = []
    seg_list = segment(pw)[0]
    if len(seg_list) == 2:
        pw_list.append(seg_list[1]+seg_list[0])
        type_list.append(0)
    return pw_list, type_list
def rule_CLRSm(mode, pw2, convert_pw1, path, count, d):
    if mode == 'Capitalize':
        candidates, types = capitalize(convert_pw1)
    elif mode == 'Leet':
        candidates, types = leet(convert_pw1)
    elif mode == 'Reverse':
        candidates, types = reverse(convert_pw1)
    else:
        assert mode == "SM"
        candidates, types = SM(convert_pw1)
    work_type = -1
    for pw, type in zip(candidates, types):
        if LD(pw, pw2)[0] < d:
            convert_pw1 = pw
            d = LD(pw, pw2)[0]
            work_type = type
    if work_type >= 0:
        path.append([mode, work_type])
        count += 1
    return convert_pw1, count, d
def DI(pw1, pw2, path, count):
    seg_list1, template1 = segment(pw1)
    seg_list2, template2 = segment(pw2)
    # 结构级
    regular_distance, distance, td_str, ti_str, hd_str, hi_str, replace_pair = LD(template1, template2)
    if len(td_str)>0:
        path.append(["TD0", td_str])
    if len(ti_str)>0:
        path.append(["TI0", ti_str])
    if len(hd_str)>0:
        path.append(["HD0", hd_str])
    if len(hi_str)>0:
        path.append(["HI0", hi_str])
    count += distance
    # 字段级
    for pair in replace_pair:
        seg1 = seg_list1[pair[0]]
        seg2 = seg_list2[pair[1]]
        regular_distance, distance, td_str, ti_str, hd_str, hi_str, replace_pair = LD(seg1, seg2)
        if len(td_str)>0:
            path.append(["TD1", td_str])
        if len(ti_str)>0:
            path.append(["TI1", ti_str])
        if len(hd_str)>0:
            path.append(["HD1", hd_str])
        if len(hi_str)>0:
            path.append(["HI1", hi_str])
        count += distance
    return count
def transfer(pw1, pw2):
    count = 0 # 修改次数
    path = [] # 修改路径
    if pw1 == pw2:
        return path, count
    d = LD(pw1, pw2)[0]  # pw1和pw2最初距离
    convert_pw1 = pw1
    # C、L、R、SM变换
    convert_pw1, count, d = rule_CLRSm("Capitalize", pw2, convert_pw1, path, count, d) # 大小写变换
    convert_pw1, count, d = rule_CLRSm("Leet", pw2, convert_pw1, path, count, d)  # 跳变变换
    convert_pw1, count, d = rule_CLRSm("Reverse", pw2, convert_pw1, path, count, d)  # 反转变换
    convert_pw1, count, d = rule_CLRSm("SM", pw2, convert_pw1, path, count, d)  # 子串移动变换
    # 经过四种变换后编辑距离仍是很大则取消变换
    if d >= 0.5:
        count = 0
        convert_pw1 = pw1
        path.clear()
    # 结构级和字段级插入删除变换
    count = DI(convert_pw1, pw2, path, count)
    return path, count


def detemine_parameter(infile, sum):
    distance_to_proportion = [0 for i in range(10)]
    fin = open(infile, 'r', encoding="UTF-8")
    while 1:
        line = fin.readline()
        if not line:
            break
        pw1 = eval(line[:-1])[0]
        pw2 = eval(line[:-1])[1]
        try:
            distance_to_proportion[transfer(pw1, pw2)[1]] += 1
        except IndexError:
            pass
            # print(pw1, pw2, transfer(pw1, pw2)[1])
    fin.close()
    last = 0
    cdf = []
    for i in range(10):
        cdf.append(distance_to_proportion[i]/sum+last)
        last = cdf[-1]
    print(cdf)
    return
def to_percent(temp, position):
    return '%1.0f' % (100 * temp) + '%'
def detemine_parameter_fig(outfile):
    plt.figure(constrained_layout=True, figsize=[8.5, 5])
    font = {'family': 'helvetica', 'size': 17}
    x = [0,1,2,3,4,5,6,7,8,9]
    y = [0.41266677168789306, 0.5249930322455558, 0.5622068998945748, 0.5800746458563067, 0.5970880845340087,
         0.6211904559940865, 0.6640148806999261, 0.724482871441901, 0.7935727009439792, 0.8574942743235219]
    plt.plot(x[1:], y[1:], label="(Financial,Financial)", color="#F07629", linestyle='--', linewidth=3)
    y = [0.42973519499277807, 0.5413577274915744, 0.567363184079602, 0.5792392874337987, 0.5916803081367357,
         0.6107141710800834, 0.6451484512919274, 0.6964403787514043, 0.7555512758786712, 0.8077611940298508]  # Non-Non
    plt.plot(x[1:], y[1:], label="(General,General)", color="#72B127", linestyle=':', linewidth=3)
    y = [0.5637697428496944, 0.6533228959361106, 0.6795764164208085, 0.6920022850207205, 0.702954429222053,
         0.7181746046444043, 0.7449459586901304, 0.7850940861555381, 0.8313436748090711, 0.8722254591766233]  # Non-Fin
    plt.plot(x[1:], y[1:], label="(Financial,General)", color="#B9B800",linestyle='-.',linewidth=3)
    plt.scatter(5, 0.6211904559940865, marker='o', s=200, facecolor='#B0E0E6')
    plt.scatter(5, 0.6107141710800834, marker='o', s=200, facecolor='#B0E0E6')
    plt.scatter(5, 0.7181746046444043, marker='o', s=200, facecolor='#B0E0E6')
    plt.legend(fontsize=17, loc='upper left', frameon=False, handlelength=5,
               labelspacing=1)
    # 设置x轴
    plt.xlabel('Thresold', font)
    plt.xticks(fontsize=17)
    plt.xlim(0.9, 9.1)
    # 设置y轴
    plt.ylabel('Reuse Rate', font)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.yticks(fontsize=17)
    plt.minorticks_on()
    plt.gca().tick_params(axis='y', which='minor', left=True)
    plt.gca().tick_params(axis='x', which='minor', bottom=False)
    plt.ylim(0.48,0.92)
    plt.savefig(outfile, bbox_inches='tight')
    plt.show()
    return
def cal_reuse(reuse, count):
    if count == 0:
        reuse['identical'] += 1
        reuse['reuse'] += 1
    elif count <= 5:
        reuse['modified'] += 1
        reuse['reuse'] += 1
    return
def cal_CLRSm(CLRSm, path):
    flag = False
    for p in path:
        if p[0] == 'Capitalize':
            CLRSm['capitalize'][p[1]] += 1
            CLRSm['capitalize'][-1] += 1
            flag = True
        elif p[0] == 'Leet':
            CLRSm['leet'][p[1]] += 1
            CLRSm['leet'][-1] += 1
            flag = True
        elif p[0] == 'Reverse':
            CLRSm['reverse'][p[1]] += 1
            CLRSm['reverse'][-1] += 1
            flag = True
        elif p[0] == 'SM':
            CLRSm['SM'][p[1]] += 1
            CLRSm['SM'][-1] += 1
            flag = True
    CLRSm['CLRSm'] += int(flag)
    return
def cal_DI(DI, path):
    flags = {'TD0': False, 'TI0': False, 'HD0': False, 'HI0': False, 'TD1': False, 'TI1': False, 'HD1': False,
             'HI1': False, 'TD': False, 'TI': False, 'HD': False, 'HI': False, 'T0': False, 'T1': False, 'H0': False,
             'H1': False, 'D0': False, 'D1': False, 'I0': False, 'I1': False, 'T': False, 'H': False, 'D': False,
             'I': False, '0': False, '1': False, 'DI': False}
    for p in path:
        if p[0] == 'TD0' or p[0] == 'TI0' or p[0] == 'HD0' or p[0] == 'HI0' or p[0] == 'TD1' or p[0] == 'TI1' \
                or p[0] == 'HD1' or p[0] == 'HI1':
            flags[p[0]] = True
            flags[p[0][0] + p[0][1]] = True
            flags[p[0][0] + p[0][2]] = True
            flags[p[0][1] + p[0][2]] = True
            flags[p[0][0]] = True
            flags[p[0][1]] = True
            flags[p[0][2]] = True
            flags['DI'] = True
    for flag in flags.keys():
        DI[flag] += int(flags[flag])
    return
def statistic(infile):
    reuse = {'reuse': 0, 'identical': 0, 'modified': 0} #重用相关计数
    CLRSm = {'capitalize': [0,0,0,0,0], 'leet': [0,0,0,0,0,0,0,0,0,0,0], 'reverse': [0,0,0], 'SM': [0,0], 'CLRSm': 0} # C、L、R和SM变换计数
    DI = {'TD0': 0, 'TI0': 0, 'HD0': 0, 'HI0': 0, 'TD1': 0, 'TI1': 0, 'HD1': 0, 'HI1': 0,
          'TD': 0, 'TI': 0, 'HD': 0, 'HI': 0, 'T0': 0, 'T1': 0, 'H0': 0, 'H1': 0, 'D0': 0, 'D1': 0, 'I0': 0, 'I1': 0,
          'T': 0, 'H': 0, 'D': 0, 'I': 0, '0': 0, '1': 0, 'DI': 0} # 插入和删除变换计数
    fin = open(infile, 'r', encoding="UTF-8")
    c = 0
    while 1:
        line = fin.readline()
        if not line:
            break
        c += 1
        pw1 = eval(line[:-1])[0]
        pw2 = eval(line[:-1])[1]
        path, count = transfer(pw1, pw2)
        cal_reuse(reuse, count)
        if count <= 4:
            cal_CLRSm(CLRSm, path)
            cal_DI(DI, path)
    print(c)
    print(reuse)
    print(CLRSm)
    print(DI)
    return
def chi_test(infile):
    fin = open(infile, 'r')
    while 1:
        line = fin.readline()
        if not line:
            break
        item = line[:-1].split('\t')[0]
        observe = np.array([[int(line[:-1].split('\t')[1]),int(line[:-1].split('\t')[2])],
                            [int(line[:-1].split('\t')[3]),int(line[:-1].split('\t')[4])]])
        chi2, p, dof, expected = chi2_contingency(observe)
        print(item, chi2, dof, p)
    fin.close()
    # 验证修改次数差异
    # fin_any = [144264, 42998, 20394, 18183]
    # non_any = [143113, 40907, 19294, 17313]
    # fin_any_pro = [i/sum(fin_any) for i in fin_any]
    # non_any_pro = [i/sum(non_any) for i in non_any]
    # print(fin_any_pro)
    # print(non_any_pro)
    # for i in range(len(fin_any)):
    #     observe = np.array([[fin_any[i],sum(fin_any)-fin_any[i]], [non_any[i],sum(non_any)-non_any[i]]])
    #     chi2, p, dof, expected = chi2_contingency(observe)
    #     print(chi2, dof, p)
    # observe = np.array([[102451,62595], [1008481, 395445]])
    # chi2, p, dof, expected = chi2_contingency(observe)
    # print(chi2, p)
    # print(observe)
    # fin_fin = [414, 444, 489, 514]
    # total_fin_fin = 1023
    # non_non = [418, 440, 463, 481]
    # total_non_non = 978
    # fin_non = [525, 563, 586, 602]
    # total_fin_non = 957
    # fin_any = [i+j for i,j in zip(fin_fin, fin_non)]
    # total_fin_any = total_fin_fin + total_fin_non
    # non_any = [i+j for i,j in zip(non_non, fin_non)]
    # total_non_any = total_non_non + total_fin_non
    # for i, j in zip(fin_any, non_any):
    #     observe = np.array([[i, total_fin_any-i], [j, total_non_any-j]])
    #     chi2, p, dof, expected = chi2_contingency(observe)
    #     print(chi2, p)
    # for i, j in zip(fin_fin, fin_non):
    #     observe = np.array([[i, total_fin_fin-i], [j, total_fin_non-j]])
    #     chi2, p, dof, expected = chi2_contingency(observe)
    #     print(chi2, p)
    return
def point(infile):
    fin = open(infile, 'r')
    lines = fin.readlines()
    fin.close()
    for index, line in enumerate(lines):
        yield 0.5 * index - 0.1, float(line[:-1].split('\t')[0]), 0.5 * index + 0.1, float(line[:-1].split('\t')[1]), \
              float(line[:-1].split('\t')[2])
def fig_reuse():
    width = 0.4
    plt.figure(constrained_layout=True, figsize=[4, 4])
    font = {'family': 'helvetica', 'size': 13}
    plt.bar([1-width/2], [0.547874659], width=width, facecolor="#F9A19A", edgecolor='white', linewidth=0.5, alpha=0.9,
            hatch="//")
    plt.bar([1+width/2], [0.550383054], width=width, facecolor="#A5D79D", edgecolor='white', linewidth=0.5, alpha=0.9,
            hatch="//")
    plt.bar([2 - width / 2], [0.160188964], width=width, facecolor="#F9A19A", edgecolor='white', linewidth=0.5,
            alpha=0.9, hatch="//")
    plt.bar([2 + width / 2], [0.157229495], width=width, facecolor="#A5D79D", edgecolor='white', linewidth=0.5,
            alpha=0.9, hatch="//")
    plt.bar([3 - width / 2], [0.708063624], width=width, facecolor="#F9A19A", edgecolor='white', linewidth=0.5,
            alpha=0.9,label="Financial")
    plt.bar([3 + width / 2], [0.707612549], width=width, facecolor="#A5D79D", edgecolor='white', linewidth=0.5,
            alpha=0.9, label="General")
    plt.text(1-width/2, 0.547874659+0.01, "54.79%", ha='center', va='center', size=8, color='black')
    plt.text(1+width/2, 0.550383054+0.01, "55.04%", ha='center', va='center', size=8, color='black')
    plt.text(2 - width / 2, 0.160188964+0.01, "16.02%", ha='center', va='center', size=8, color='black')
    plt.text(2 + width / 2, 0.157229495+0.01, "15.72%", ha='center', va='center', size=8, color='black')
    plt.text(3 - width / 2, 0.708063624+0.01, "70.81%", ha='center', va='center', size=8, color='black')
    plt.text(3 + width / 2, 0.707612549+0.01, "70.76%", ha='center', va='center', size=8, color='black')
    plt.legend(fontsize=13, loc='upper left', frameon=False, handlelength=4, labelspacing=0.3)
    # 设置x轴
    plt.xlim(0.55,3.45)
    plt.xticks([1, 2, 3], ["Identical", "Modified", "Any of above"], fontsize=13)
    # 设置y轴
    plt.ylim(0, 0.76)
    plt.yticks(fontsize=13)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.10))
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    plt.ylabel('Proportion', font)
    # 刻度朝内
    plt.tick_params(top='in', right='in', which='minor', direction='in')
    plt.tick_params(top='in', right='in', direction='in')
    plt.savefig("Reuse.pdf", bbox_inches='tight')
    plt.savefig("Reuse.png", bbox_inches='tight')
    plt.show()
    return
def fig_CLRSm():
    width = 0.4
    plt.figure(constrained_layout=True, figsize=[6.7, 4])
    font = {'family': 'helvetica', 'size': 13}
    plt.bar([1-width/2], [0.159390766], width=width, facecolor="#F9A19A", edgecolor='white', linewidth=0.5, alpha=0.9, hatch="//")
    plt.bar([1+width/2], [0.151966921], width=width, facecolor="#A5D79D", edgecolor='white', linewidth=0.5, alpha=0.9, hatch="//")
    plt.bar([2 - width / 2], [0.008148584], width=width, facecolor="#F9A19A", edgecolor='white', linewidth=0.5,
            alpha=0.9,label="Financial")
    plt.bar([2 + width / 2], [0.008204509], width=width, facecolor="#A5D79D", edgecolor='white', linewidth=0.5,
            alpha=0.9,label="General")
    plt.bar([3 - width / 2], [0.008478825], width=width, facecolor="#F9A19A", edgecolor='white', linewidth=0.5,
            alpha=0.9)
    plt.bar([3 + width / 2], [0.007992464], width=width, facecolor="#A5D79D", edgecolor='white', linewidth=0.5,
            alpha=0.9)
    plt.bar([4 - width / 2], [0.004460236], width=width, facecolor="#F9A19A", edgecolor='white', linewidth=0.5,
            alpha=0.9)
    plt.bar([4 + width / 2], [0.004432555], width=width, facecolor="#A5D79D", edgecolor='white', linewidth=0.5,
            alpha=0.9)
    plt.bar([5 - width / 2], [0.177943915], width=width, facecolor="#F9A19A", edgecolor='white', linewidth=0.5,
            alpha=0.9, hatch="//")
    plt.bar([5 + width / 2], [0.170047832], width=width, facecolor="#A5D79D", edgecolor='white', linewidth=0.5,
            alpha=0.9, hatch="//")
    plt.text(1 - width / 2, 0.159390766+0.003, "15.94%", ha='center', va='center', size=8, color='black')
    plt.text(1 + width / 2, 0.151966921+0.003, "15.20%", ha='center', va='center', size=8, color='black')
    plt.text(2 - width / 2, 0.008148584+0.003, "0.81%", ha='center', va='center', size=8, color='black')
    plt.text(2 + width / 2, 0.008204509+0.003, "0.82%", ha='center', va='center', size=8, color='black')
    plt.text(3 - width / 2, 0.008478825+0.003, "0.85%", ha='center', va='center', size=8, color='black')
    plt.text(3 + width / 2, 0.007992464+0.003, "0.80%", ha='center', va='center', size=8, color='black')
    plt.text(4 - width / 2, 0.004460236+0.003, "0.45%", ha='center', va='center', size=8, color='black')
    plt.text(4 + width / 2, 0.004432555+0.003, "0.44%", ha='center', va='center', size=8, color='black')
    plt.text(5 - width / 2, 0.177943915+0.003, "17.79%", ha='center', va='center', size=8, color='black')
    plt.text(5 + width / 2, 0.170047832+0.003, "17.00%", ha='center', va='center', size=8, color='black')
    plt.legend(fontsize=13, loc='upper center', frameon=False, handlelength=4, labelspacing=0.3)
    # 设置x轴
    plt.xlim(0.55, 5.45)
    plt.xticks([1, 2, 3, 4, 5], ["Capitalization", "Leet", "Reverse", "SM", "Any of above"], fontsize=13)
    # 设置y轴
    plt.ylim(0, 0.20)
    plt.yticks(fontsize=13)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.02))
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    plt.ylabel('Proportion', font)
    # 刻度朝内
    plt.tick_params(top='in', right='in', which='minor', direction='in')
    plt.tick_params(top='in', right='in', direction='in')
    plt.savefig("CLRSm.pdf", bbox_inches='tight')
    plt.savefig("CLRSm.png", bbox_inches='tight')
    plt.show()
    return
def fig_DI():
    width = 0.4
    plt.figure(constrained_layout=True, figsize=[4, 4])
    font = {'family': 'helvetica', 'size': 13}
    plt.bar([1 - width / 2], [0.213044897], width=width, facecolor="#F9A19A", edgecolor='white', linewidth=0.5,
            alpha=0.9, hatch="//")
    plt.bar([1 + width / 2], [0.334884252], width=width, facecolor="#A5D79D", edgecolor='white', linewidth=0.5,
            alpha=0.9, hatch="//")
    plt.bar([2 - width / 2], [0.373012589], width=width, facecolor="#F9A19A", edgecolor='white', linewidth=0.5,
            alpha=0.9, hatch="//")
    plt.bar([2 + width / 2], [0.232951788], width=width, facecolor="#A5D79D", edgecolor='white', linewidth=0.5,
            alpha=0.9, hatch="//")
    plt.bar([3 - width / 2], [0.546969745], width=width, facecolor="#F9A19A", edgecolor='white', linewidth=0.5,
            alpha=0.9, hatch="//")
    plt.bar([3 + width / 2], [0.530153202], width=width, facecolor="#A5D79D", edgecolor='white', linewidth=0.5,
            alpha=0.9, hatch="//")
    plt.bar([10], [10], width=width, facecolor="#F9A19A", edgecolor='white', linewidth=0.5, alpha=0.9, label="Financial")
    plt.bar([11], [12], width=width, facecolor="#A5D79D", edgecolor='white', linewidth=0.5,
            alpha=0.9, label="General")
    plt.text(1 - width / 2, 0.213044897+0.01, "21.30%", ha='center', va='center', size=8, color='black')
    plt.text(1 + width / 2, 0.334884252+0.01, "33.49%", ha='center', va='center', size=8, color='black')
    plt.text(2 - width / 2, 0.373012589+0.01, "37.30%", ha='center', va='center', size=8, color='black')
    plt.text(2 + width / 2, 0.232951788+0.01, "23.30%", ha='center', va='center', size=8, color='black')
    plt.text(3 - width / 2, 0.546969745+0.01, "54.70%", ha='center', va='center', size=8, color='black')
    plt.text(3 + width / 2, 0.530153202+0.01, "53.02%", ha='center', va='center', size=8, color='black')
    plt.legend(fontsize=13, loc='upper left', frameon=False, handlelength=4, labelspacing=0.3)
    # 设置x轴
    plt.xlim(0.55, 3.45)
    plt.xticks([1, 2, 3], ["Deletion", "Insertion", "Any of above"], fontsize=13)
    # 设置y轴
    plt.ylim(0, 0.65)
    plt.yticks(fontsize=13)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    plt.ylabel('Proportion', font)
    # 刻度朝内
    plt.tick_params(top='in', right='in', which='minor', direction='in')
    plt.tick_params(top='in', right='in', direction='in')
    plt.savefig("DI.pdf", bbox_inches='tight')
    plt.savefig("DI.png", bbox_inches='tight')
    plt.show()
def targuessII_fig(outfile):
    plt.figure(constrained_layout=True, figsize=[8.5, 5])
    font = {'family': 'helvetica', 'size': 17}
    x = [1, 10, 100, 1000]
    y = [0.4046920821114369, 0.4340175953079179, 0.4780058651026393, 0.5024437927663734] #fin_fin
    plt.plot(x, y, label="financial -> financial", color="#F07629", linestyle='--', linewidth=3)
    y = [0.54858934169279, 0.5882967607105538, 0.6123301985370951, 0.6290491118077325]  # fin_non
    plt.plot(x, y, label="general -> financial / financial -> general", color="#B9B800", linestyle=':', linewidth=3)
    y = [0.47424242424242424, 0.5085858585858586, 0.5429292929292929, 0.5636363636363636]  # any_fin
    plt.plot(x, y, label="any -> financial", color="#F07629", linestyle='-.', linewidth=3)
    y = [0.4274028629856851, 0.4498977505112474, 0.4734151329243354, 0.4918200408997955] # non_non
    plt.plot(x, y, label="general -> general", color="#72B127", linestyle='--', linewidth=3)
    y = [0.4873385012919897, 0.5183462532299742, 0.5421188630490956, 0.5596899224806201] # any_non
    plt.plot(x, y, label="any -> general", color="#72B127", linestyle='-.', linewidth=3)
    plt.legend(fontsize=17, loc='upper left', frameon=False, handlelength=5, labelspacing=0.5)
    # 设置x轴
    plt.xlabel('Guess Number', font)
    plt.ylabel('Fraction of Cracked Passwords', font)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.xscale('symlog')
    plt.xlim(0.9, 1000 + 0.1)
    plt.ylim(0.38, 0.90)
    plt.yticks(fontsize=17)
    plt.xticks([1, 10, 100, 1000], fontsize=17)
    plt.minorticks_on()
    plt.tick_params(top='in', right='in', which='minor', direction='in')
    plt.tick_params(top='in', right='in', direction='in')
    plt.savefig(outfile, bbox_inches='tight')
    plt.show()
    return

def fig_capitalization():
    width = 0.4
    plt.figure(constrained_layout=True, figsize=[6.7, 4])
    font = {'family': 'helvetica', 'size': 13}
    plt.bar([1-width/2], [0.159390766], width=width, facecolor="#F9A19A", edgecolor='white', linewidth=0.5, alpha=0.9, hatch="//")
    plt.bar([1+width/2], [0.151966921], width=width, facecolor="#A5D79D", edgecolor='white', linewidth=0.5, alpha=0.9, hatch="//")
    plt.bar([2 - width / 2], [0.008148584], width=width, facecolor="#F9A19A", edgecolor='white', linewidth=0.5,
            alpha=0.9,label="Financial")
    plt.bar([2 + width / 2], [0.008204509], width=width, facecolor="#A5D79D", edgecolor='white', linewidth=0.5,
            alpha=0.9,label="General")
    plt.bar([3 - width / 2], [0.008478825], width=width, facecolor="#F9A19A", edgecolor='white', linewidth=0.5,
            alpha=0.9)
    plt.bar([3 + width / 2], [0.007992464], width=width, facecolor="#A5D79D", edgecolor='white', linewidth=0.5,
            alpha=0.9)
    plt.bar([4 - width / 2], [0.004460236], width=width, facecolor="#F9A19A", edgecolor='white', linewidth=0.5,
            alpha=0.9)
    plt.bar([4 + width / 2], [0.004432555], width=width, facecolor="#A5D79D", edgecolor='white', linewidth=0.5,
            alpha=0.9)
    plt.bar([5 - width / 2], [0.177943915], width=width, facecolor="#F9A19A", edgecolor='white', linewidth=0.5,
            alpha=0.9, hatch="//")
    plt.bar([5 + width / 2], [0.170047832], width=width, facecolor="#A5D79D", edgecolor='white', linewidth=0.5,
            alpha=0.9, hatch="//")
    plt.text(1 - width / 2, 0.159390766+0.003, "15.94%", ha='center', va='center', size=8, color='black')
    plt.text(1 + width / 2, 0.151966921+0.003, "15.20%", ha='center', va='center', size=8, color='black')
    plt.text(2 - width / 2, 0.008148584+0.003, "0.81%", ha='center', va='center', size=8, color='black')
    plt.text(2 + width / 2, 0.008204509+0.003, "0.82%", ha='center', va='center', size=8, color='black')
    plt.text(3 - width / 2, 0.008478825+0.003, "0.85%", ha='center', va='center', size=8, color='black')
    plt.text(3 + width / 2, 0.007992464+0.003, "0.80%", ha='center', va='center', size=8, color='black')
    plt.text(4 - width / 2, 0.004460236+0.003, "0.45%", ha='center', va='center', size=8, color='black')
    plt.text(4 + width / 2, 0.004432555+0.003, "0.44%", ha='center', va='center', size=8, color='black')
    plt.text(5 - width / 2, 0.177943915+0.003, "17.79%", ha='center', va='center', size=8, color='black')
    plt.text(5 + width / 2, 0.170047832+0.003, "17.00%", ha='center', va='center', size=8, color='black')
    plt.legend(fontsize=13, loc='upper center', frameon=False, handlelength=4, labelspacing=0.3)
    # 设置x轴
    plt.xlim(0.55, 5.45)
    plt.xticks([1, 2, 3, 4, 5], ["Capitalization", "Leet", "Reverse", "SM", "Any of above"], fontsize=13)
    # 设置y轴
    plt.ylim(0, 0.20)
    plt.yticks(fontsize=13)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.02))
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    plt.ylabel('Proportion', font)
    # 刻度朝内
    plt.tick_params(top='in', right='in', which='minor', direction='in')
    plt.tick_params(top='in', right='in', direction='in')
    plt.savefig("CLRSm.pdf", bbox_inches='tight')
    plt.savefig("CLRSm.png", bbox_inches='tight')
    plt.show()
    return

def draw_fig():
    # 示例数据
    plt.figure(constrained_layout=True, figsize=[4, 2.5])
    values = [0.6388, 0.1904, 0.0903, 0.0805]
    plt.barh([0.24, 0.48, 0.72, 0.96], values[::-1], height=0.2, color='#F9A19A')
    plt.yticks([0.24, 0.48, 0.72, 0.96], ['one', 'two', 'three', 'four'][::-1], fontsize=15)
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], ["0%", "10%","20%", "30%", "40%", "50%", "60%"], fontsize=15)
    plt.text(0.24, 0.92, "63.88%", fontsize=15)
    plt.text(0.20, 0.68, "19.04%", fontsize=15)
    plt.text(0.10, 0.44, "9.03%", fontsize=15)
    plt.text(0.10, 0.20, "8.05%", fontsize=15)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    # plt.xlabel('数值')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    # plt.grid(axis='x', alpha=0.3)
    plt.show()

def draw_fig2():
    # 示例数据
    plt.figure(constrained_layout=True, figsize=[4, 2.5])
    values = [0.6487, 0.1854, 0.0875, 0.0785]
    plt.barh([0.24, 0.48, 0.72, 0.96], values[::-1], height=0.2, color='#A5D79D')
    plt.yticks([0.24, 0.48, 0.72, 0.96], ['one', 'two', 'three', 'four'][::-1], fontsize=15)
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], ["0%", "10%","20%", "30%", "40%", "50%", "60%"], fontsize=15)
    plt.text(0.24, 0.92, "64.87%", fontsize=15)
    plt.text(0.19, 0.68, "18.54%", fontsize=15)
    plt.text(0.10, 0.44, "8.75%", fontsize=15)
    plt.text(0.10, 0.20, "7.85%", fontsize=15)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    # plt.xlabel('数值')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    # plt.grid(axis='x', alpha=0.3)
    plt.show()

if __name__ == '__main__':
    draw_fig2()
    # union("reuse/merge-sort.txt", "reuse/union.txt") # 账户合并，输入文件处理成email'\t'password'\t'website格式
    # pairs("reuse/union.txt", "reuse/Fin-Fin.txt", "reuse/Non-Non.txt",
    #       "reuse/Non-Fin.txt", "Fin-Non.txt") # 重用口令对 165470 985579 2285440 2285440
    # statistics("Distinct_new.txt")
    # detemine_parameter("Fin-Fin.txt", 165046)
    # detemine_parameter("Non-Non.txt", 155775)
    # detemine_parameter("Fin-Non.txt", 1403926)  # 找口令重用的距离阈值
    # detemine_parameter_fig("Threshold.png")
    # detemine_parameter_fig("Threshold.pdf")  # 画图找口令重用的距离阈值
    # statistic("Data2024/Reuse/Fin-Fin.txt")
    # statistic("Data2024/Reuse/Fin-Non.txt")
    # statistic("Data2024/Reuse/Non-Non.txt")
    # statistic("Data2024/Reuse/Non-Fin.txt") # 重用数据统计
    # chi_test("Result2024/Reuse/Statistics.txt") # 重用数据卡方检验
    # fig_reuse()
    # fig_CLRSm()
    # fig_DI()
    # targuessII_fig("TarguessII.png")
    # targuessII_fig("TarguessII.pdf")








