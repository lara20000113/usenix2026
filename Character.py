import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from scipy import stats
import numpy as np

def to_percent(temp, position):
    return '%1.0f' % (100 * float(temp)) + '%'
def drawFig():
    one = [0.2960, 0.4709, 0.6085883541352592, 0.5927217375183325, 0.4528570604873558]
    two = [0.5815, 0.4512, 0.3739919730349192, 0.36717812982278025, 0.4930640911950425]
    three = [0.1107, 0.0711,0.016858516591873225, 0.03838423281171526, 0.050886609521054664]
    four = [0.0119, 0.0068, 0.0005611562379483224, 0.00171589984717199, 0.00319223879654704]
    # 柱体底部
    bottom3 = [i + j for i, j in zip(one, two)]
    bottom4 = [i + j + k for i, j, k in zip(one, two, three)]
    plt.figure(constrained_layout=True, figsize=[7,3]) # 画布大小
    width = 0.2  # 柱子的宽度
    font = {'family': 'helvetica', 'size': 15} # 字体
    plt.barh([i*0.24 for i in range(5)], one, height=width, facecolor="#F9A19A", edgecolor='white', linewidth=0.5, alpha=0.9,
            label="one")
    plt.barh([i*0.24 for i in range(5)], two, height=width, facecolor="#FFCB9D", edgecolor='white', left=one, linewidth=0.5,
             alpha=0.9, label="two")
    plt.barh([i*0.24 for i in range(5)], three, height=width, facecolor="#FFF59D", edgecolor='white', left=bottom3, linewidth=0.5,
             alpha=0.9, label="three")
    plt.barh([i*0.24 for i in range(5)], four, height=width, facecolor="#A5D79D", edgecolor='white', left=bottom4, linewidth=0.5,
             alpha=0.9, label="four")
    plt.text(0.10, -0.03, "29.60%", fontsize=12, color="black")
    plt.text(0.50, -0.03, "58.15%", fontsize=12, color="black")
    plt.text(0.875, -0.03, "11.07%", fontsize=12, color="black")
    plt.text(0.20, 0.21, "47.09%", fontsize=12, color="black")
    plt.text(0.65, 0.21, "45.12%", fontsize=12, color="black")
    plt.text(0.93, 0.21, "7.71%", fontsize=12, color="black")
    plt.text(0.25, 0.45, "60.86%", fontsize=12, color="black")
    plt.text(0.74, 0.45, "37.40%", fontsize=12, color="black")
    plt.text(0.96, 0.45, "1.69%", fontsize=12, color="black")
    plt.text(0.23, 0.69, "59.27%", fontsize=12, color="black")
    plt.text(0.73, 0.69, "36.72%", fontsize=12, color="black")
    plt.text(0.96, 0.69, "3.84%", fontsize=12, color="black")
    plt.text(0.20, 0.93, "45.26%", fontsize=12, color="black")
    plt.text(0.63, 0.93, "49.31%", fontsize=12, color="black")
    plt.text(0.95, 0.93, "5.09%", fontsize=12, color="black")
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
    plt.savefig("Type.pdf", bbox_inches='tight')
    plt.savefig("Type.png", bbox_inches='tight')
    plt.show()
    return

def chi_squared_test(infile):
    lines = open(infile, 'r').readlines()
    for line in lines:
        observe = np.array([[int(line[:-1].split('\t')[1]), int(line[:-1].split('\t')[2])],
                            [int(line[:-1].split('\t')[3]), int(line[:-1].split('\t')[4])]])
        # print(observe)
        chi2, p, dof, expected = stats.chi2_contingency(observe)
        print(line[:-1].split('\t')[0], chi2, dof, p,
              int(line[:-1].split('\t')[1]) / (int(line[:-1].split('\t')[1])+int(line[:-1].split('\t')[2])),
              int(line[:-1].split('\t')[3]) / (int(line[:-1].split('\t')[3]) + int(line[:-1].split('\t')[4])))
    return

if __name__ == '__main__':
    chi_squared_test("result/char/Data_for_Chi_Squared.txt")
    # drawFig()