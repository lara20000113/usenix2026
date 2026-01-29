from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from read_file import read_file, read_file_format_pw_details_semantics
from utils import (load_pickle, save_pickle, weighted_mean, chi_squared,
                   DATASET2SIZE, DATASETS, CATEGORY2DATASETS, CATEGORIES)

# 语义模式
patterns = ["single_character_repeat", "segment_repeat", "sequence_downup", "keybord", "palindrome",
            "YYYYMMDD", "YYMMDD", "MMDD", "YYYY",
            "love_related", "website_related", "English_word",
            "firstname", "lastname", "fullname"]
categories = ["simple", "date", "word", "name"]
pattern2category = {"single_character_repeat": "simple", "segment_repeat": "simple", "sequence_downup": "simple",
                    "keybord": "simple", "palindrome": "simple",
                    "YYYYMMDD": "date", "YYMMDD": "date", "MMDD": "date", "YYYY": "date",
                    "love_related": "word", "website_related": "word", "English_word": "word",
                    "firstname": "name", "lastname": "name", "fullname": "name"}

# 自定义类
class model_manage:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
    def get(self):
        if not self.model:
            self.model = load_pickle(self.model_path)
        return self.model
semantic_pattern_datasets_model = model_manage("result/semantic/semantic_pattern_datasets.pkl")
semantic_pattern_categories_model = model_manage("result/semantic/semantic_pattern_categories.pkl")
semantic_coverage_datasets_model = model_manage("result/semantic/semantic_coverage_datasets.pkl")
semantic_coverage_categories_model = model_manage("result/semantic/semantic_coverage_categories.pkl")

class Node:
    def __init__(self, char=""):
        self.char = char
        self.next = {}
class Trie:
    def __init__(self, file):
        self.root = Node()
        self.construct(file)
    def construct(self, file):
        read_file_class = read_file()
        g = read_file_class.read(file)
        while 1:
            try:
                str = next(g)
                self.insert(str)
            except StopIteration:
                return
    def insert(self, line):
        line += '\n'
        node = self.root
        for c in line:
            if c not in node.next.keys():
                node.next[c] = Node(c)
            node = node.next[c]
        return
    def search(self, pw):
        node = self.root
        for c in pw:
            if c not in node.next.keys():
                return False
            else:
                node = node.next[c]
        return '\n' in node.next.keys()
class parse:
    def __init__(self,
                 keyboardFile="./dictionary/Keyboard.txt",
                 YYYYMMDDFile="./dictionary/Date_YYYYMMDD.txt",
                 YYMMDDFile="./dictionary/Date_YYMMDD.txt",
                 MMDDFile="./dictionary/Date_MMDD.txt",
                 YYYYFile="./dictionary/Date_YYYY.txt",
                 firstname_file = "./dictionary/English_firstname_5.txt",
                 lastname_file = "./dictionary/English_lastname_5.txt",
                 fullname_file = "./dictionary/English_fullname_5.txt",
                 English_word_file="./dictionary/English_Word.txt"):
        self.keyBoard_trie = Trie(keyboardFile)
        self.YYYYMMDD_trie = Trie(YYYYMMDDFile)
        self.YYMMDD_trie = Trie(YYMMDDFile)
        self.MMDD_trie = Trie(MMDDFile)
        self.YYYY_trie = Trie(YYYYFile)
        self.love_related_list = ["love"]
        self.website_related_list = []
        self.English_word_trie = Trie(English_word_file)
        self.firstname_trie = Trie(firstname_file)
        self.lastname_trie = Trie(lastname_file)
        self.fullname_trie = Trie(fullname_file)
    def construct_list(self, file):
        fin = open(file, 'r')
        list = [line[:-1] for line in fin.readlines()]
        fin.close()
        return list
    # 判断字符串是否是重复序列
    def belong_single_character_repeat(self, string):
        for i in range(1,len(string)):
            if string[i]!=string[i-1]:
                return False
        return len(string)>=3
    # 判断字符串是否是段重复序列
    def belong_segment_repeat(self, string):
        for l in range(2, len(string)):
            if len(string)%l!=0:
                continue
            index = 0
            while index+2*l<=len(string) and string[index:index+l]==string[index+l:index+2*l]:
                index += l
            if index+2*l>len(string):
                return True
        return False
    # 判断字符串是否是顺序序列
    def belong_sequence_downup(self, string):
        flag1 = True
        ordList = [ord(c) for c in string]
        for i in range(1, len(string)):
            if ordList[i]-ordList[i-1] != 1:
                flag1 = False
        flag2 = True
        for i in range(1, len(string)):
            if ordList[i]-ordList[i-1] != -1:
                flag2 = False
        return (flag1 or flag2) and len(string)>=3
    # 判断字符串是否是键盘序列
    def belong_keybord(self, string):
        return self.keyBoard_trie.search(string)
    # 判断字符串是否是回文序列
    def belong_palindrome(self, string):
        for i,j in zip(range(0,len(string),1), range(len(string)-1,-1,-1)):
            if i>j:
                break
            if string[i]!=string[j]:
                return False
        return len(string)>=4
    # 判断字符串是否是YYYYMMDD
    def belong_YYYYMMDD(self, string):
        # return string in self.YYYYMMDD_list
        return self.YYYYMMDD_trie.search(string)
    # 判断字符串是否是YYMMDD
    def belong_YYMMDD(self, string):
        # return string in self.YYMMDD_list
        return self.YYMMDD_trie.search(string)
    # 判断字符串是否是MMDD
    def belong_MMDD(self, string):
        # return string in self.MMDD_list
        return self.MMDD_trie.search(string)
    # 判断字符串是否是YYYY
    def belong_YYYY(self, string):
        # return string in self.YYYY_list
        return self.YYYY_trie.search(string)
    # 判断字符串是否与love相关
    def belong_love_related(self, string):
        return string in self.love_related_list
    # 判断字符串是否与website相关
    def belong_website_related(self, string):
        return string in self.website_related_list
    # 判断字符串是否是英语词汇
    def belong_English_word(self, string):
        # return string in self.English_word_list
        return self.English_word_trie.search(string)
    # 判断字符串是否是firstname
    def belong_firstname(self, string):
        # return string in self.firstname_list
        return self.firstname_trie.search(string)
    # 判断字符串是否是lastname
    def belong_lastname(self, string):
        # return string in self.lastname_list
        return self.lastname_trie.search(string)
    # 判断字符串是否是fullname
    def belong_fullname(self, string):
        return self.fullname_trie.search(string)
    # 判断字符串是哪种语义模式
    def mode(self, string):
        if self.belong_single_character_repeat(string):
            return "single_character_repeat"
        if self.belong_segment_repeat(string):
            return "segment_repeat"
        if self.belong_sequence_downup(string):
            return "sequence_downup"
        if self.belong_keybord(string):
            return "keybord"
        if self.belong_palindrome(string):
            return "palindrome"
        if self.belong_YYYYMMDD(string):
            return "YYYYMMDD"
        if self.belong_YYMMDD(string):
            return "YYMMDD"
        if self.belong_MMDD(string):
            return "MMDD"
        if self.belong_YYYY(string):
            return "YYYY"
        if self.belong_love_related(string):
            return "love_related"
        if self.belong_website_related(string):
            return "website_related"
        if self.belong_English_word(string):
            return "English_word"
        if self.belong_firstname(string):
            return "firstname"
        if self.belong_lastname(string):
            return "lastname"
        if self.belong_fullname(string):
            return "fullname"
        return None
    # 解析字符串语义段
    def sysn(self, string):
        n = len(string)
        segments = [None for i in range(n)]
        lengths = [0 for i in range(n+1)]
        nums = [0 for i in range(n+1)]
        for start in range(n-1, -1, -1):
            lengths[start] = lengths[start+1]
            nums[start] = nums[start+1]
            for end in range(start, n):
                if self.mode(string[start:end+1]):
                    if end+1-start+lengths[end+1]>lengths[start]:
                        segments[start] = string[start:end+1]
                        lengths[start] = end+1-start+lengths[end+1]
                        nums[start] = 1+nums[end+1]
                    elif end+1-start+lengths[end+1]==lengths[start] and nums[end+1]+1<nums[start]:
                        segments[start] = string[start:end + 1]
                        nums[start] = 1 + nums[end + 1]
        semantics = []
        details = []
        index = 0
        while index<n:
            start = index
            while index<n and not segments[index]:
                index += 1
            if index>start:
                semantics.append(string[start:index])
                details.append(string[start:index])
            if index<n:
                semantics.append(segments[index])
                details.append(self.mode(segments[index]))
                index += len(segments[index])
        return semantics, details
    # 解析文件语义段
    def parser_file(self, infile, outfile):
        read_file_class = read_file()
        g = read_file_class.read(infile)
        fout = open(outfile, 'w', encoding="UTF-8")
        while 1:
            try:
                pw = next(g)
                details, semantics = self.sysn(pw.lower())
                fout.write(pw+'\t'+str(details)+'\t'+str(semantics)+'\n')
            except StopIteration:
                fout.close()
                return

def _semantic_pattern_dataset(dataset):
    semantic_pattern = {pattern: 0 for pattern in patterns + categories}
    read_file_class = read_file_format_pw_details_semantics()
    g = read_file_class.read("semantic/{}.txt".format(dataset))
    while 1:
        try:
            pw, details, semantics = next(g)
            semantics_set, categories_set = set([]), set([])
            for detail, semantic in zip(details, semantics):
                if detail != semantic:
                    semantics_set.add(semantic)
                    categories_set.add(pattern2category[semantic])
            for semantic in semantics_set:
                semantic_pattern[semantic] += 1
            for category in categories_set:
                semantic_pattern[category] += 1
        except StopIteration:
            return {pattern: semantic_pattern[pattern]/DATASET2SIZE[dataset] for pattern in patterns}
def semantic_pattern_datasets():
    dataset2semantic_pattern = {dataset: _semantic_pattern_dataset(dataset) for dataset in DATASETS}
    save_pickle(semantic_pattern_datasets_model.model_path, dataset2semantic_pattern)
    return
def _semantic_pattern_category(category):
    datasets = CATEGORY2DATASETS[category]
    semantic_pattern = {pattern: 0 for pattern in patterns}
    for pattern in patterns:
        p = [semantic_pattern_datasets_model.get()[dataset][pattern] for dataset in datasets]
        q = [DATASET2SIZE[dataset] for dataset in datasets]
        semantic_pattern[pattern] = weighted_mean(p, q)
    return semantic_pattern
def semantic_pattern_categories():
    category2semantic_pattern = {category: _semantic_pattern_category(category) for category in CATEGORIES}
    print(category2semantic_pattern)
    save_pickle(semantic_pattern_categories_model.model_path, category2semantic_pattern)
    return
def _semantic_coverage_dataset(dataset):
    semantic_coverage = {"full": 0, "partial": 0, "none": 0}
    read_file_class = read_file_format_pw_details_semantics()
    g = read_file_class.read("semantic/{}.txt".format(dataset))
    while 1:
        try:
            pw, details, semantics = next(g)
            semantic_length = 0
            for detail, semantic in zip(details, semantics):
                if detail != semantic:
                    semantic_length += len(detail)
            if semantic_length == 0:
                semantic_coverage["none"] += 1
            elif semantic_length < len(pw):
                semantic_coverage["partial"] += 1
            else:
                assert semantic_length == len(pw)
                semantic_coverage['full'] += 1
        except StopIteration:
            return {key: semantic_coverage[key]/DATASET2SIZE[dataset] for key in semantic_coverage.keys()}
def semantic_coverage_datasets():
    dataset2semantic_coverage = {dataset: _semantic_coverage_dataset(dataset) for dataset in DATASETS}
    save_pickle(semantic_coverage_datasets_model.model_path, dataset2semantic_coverage)
    return
def _semantic_coverage_category(category):
    datasets = CATEGORY2DATASETS[category]
    semantic_coverage = {"full": 0, "partial": 0, "none": 0}
    for key in semantic_coverage.keys():
        p = [semantic_coverage_datasets_model.get()[dataset][key] for dataset in datasets]
        q = [DATASET2SIZE[dataset] for dataset in datasets]
        semantic_coverage[key] = weighted_mean(p, q)
    return semantic_coverage
def semantic_coverage_categories():
    category2semantic_coverage = {category: _semantic_coverage_category(category) for category in CATEGORIES}
    save_pickle(semantic_coverage_categories_model.model_path, category2semantic_coverage)
    return

def to_percent(temp, position):
    return '%1.0f' % (100 * float(temp)) + '%'
def draw_semantic_coverage():
    one = [0.3646, 0.3603, 0.3719, 0.3439, 0.3770]
    two = [0.4925, 0.4892, 0.3767, 0.4165, 0.4295]
    three = [0.1429, 0.1504, 0.2514, 0.2396, 0.1934]
    # 柱体底部
    bottom3 = [i + j for i, j in zip(one, two)]
    bottom4 = [i + j + k for i, j, k in zip(one, two, three)]
    plt.figure(constrained_layout=True, figsize=[7,3]) # 画布大小
    width = 0.2  # 柱子的宽度
    font = {'family': 'helvetica', 'size': 15} # 字体
    plt.barh([i*0.24 for i in range(5)], one, height=width, facecolor="#A5D79D", edgecolor='white', linewidth=0.5, alpha=0.9,
            label="None")
    plt.barh([i*0.24 for i in range(5)], two, height=width, facecolor="#FFCB9D", edgecolor='white', left=one, linewidth=0.5,
             alpha=0.9, label="Partial")
    plt.barh([i*0.24 for i in range(5)], three, height=width, facecolor="#F9A19A", edgecolor='white', left=bottom3, linewidth=0.5,
             alpha=0.9, label="Full")
    plt.text(0.13, -0.03, "36.46%", fontsize=12, color="black")
    plt.text(0.55, -0.03, "49.25%", fontsize=12, color="black")
    plt.text(0.875, -0.03, "14.29%", fontsize=12, color="black")
    plt.text(0.135, 0.21, "36.03%", fontsize=12, color="black")
    plt.text(0.57, 0.21, "48.92%", fontsize=12, color="black")
    plt.text(0.86, 0.21, "15.04%", fontsize=12, color="black")
    plt.text(0.14, 0.45, "37.19%", fontsize=12, color="black")
    plt.text(0.47, 0.45, "37.67%", fontsize=12, color="black")
    plt.text(0.83, 0.45, "25.14%", fontsize=12, color="black")
    plt.text(0.13, 0.69, "34.39%", fontsize=12, color="black")
    plt.text(0.48, 0.69, "41.65%", fontsize=12, color="black")
    plt.text(0.83, 0.69, "23.96%", fontsize=12, color="black")
    plt.text(0.15, 0.93, "37.70%", fontsize=12, color="black")
    plt.text(0.53, 0.93, "42.95%", fontsize=12, color="black")
    plt.text(0.84, 0.93, "19.34%", fontsize=12, color="black")
    # 上边界和右边界去掉
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    # Y轴设置
    plt.yticks((0, 0.24, 0.48, 0.72, 0.96), ("Financial", "Social", "Email", "Forum", "Content"), fontsize=15)
    plt.ylim(-0.12,1.1)
    # X轴设置
    plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(0.10))
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    plt.xticks(fontsize=15)
    plt.xlim(0, 1.0)
    # 图例设置
    plt.legend(loc='upper center', ncol=4, fontsize=15, frameon=False, bbox_to_anchor=(0.45, 1.1, 0.1, 0.1))
    plt.show()
    return


if __name__ == "__main__":
    # parser = parse()
    # for dataset in DATASETS:
    #     parser.website_related_list.clear()
    #     parser.website_related_list.append(dataset.lower())
    #     parser.parser_file("data/"+dataset+"txt", "result/semantic/"+dataset+"txt")
    # semantic_pattern_categories()
    # semantic_coverage_categories()
    chi_squared(semantic_coverage_datasets_model.get())
    draw_semantic_coverage()


