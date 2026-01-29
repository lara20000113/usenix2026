import math
from utils import load_pickle, save_pickle, weighted_mean, DATASETS, DATASET2SIZE, CATEGORIES, CATEGORY2DATASETS

# 自定义类
class model_manage:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
    def get(self):
        if not self.model:
            self.model = load_pickle(self.model_path)
        return self.model
metrics_datasets_model = model_manage("result/metrics/metrics_datasets.pkl")
metrics_categories_model = model_manage("result/metrics/metrics_categories.pkl")

def getP(inFile):
    p = []
    fin = open(inFile, 'r', encoding="UTF-8")
    while 1:
        line = fin.readline()
        if not line:
            break
        pro = float(line[:-1].split('\t')[-1])
        p.append(pro)
    fin.close()
    p.sort(reverse=True)
    return p
def Hinf():
    return -math.log(p[0], 2)
def H1():
    h1 = 0
    for pro in p:
        h1 = h1 - pro * math.log(pro, 2)
    return h1
def LambdaBeta(beta):
    lambdaBeta = 0
    for i in range(beta):
        lambdaBeta += p[i]
    return lambdaBeta
def LambdaAdvBeta(beta):
    return math.log(beta/LambdaBeta(beta), 2)
def MuAlpha(alpha):
    muAlpha = 0
    s = 0
    for pro in p:
        s += pro
        muAlpha += 1
        if s >= alpha:
            break
    return muAlpha
def MuAdvAlpha(alpha):
    muAlpha = MuAlpha(alpha)
    return math.log(muAlpha / LambdaBeta(muAlpha), 2)
def GAlpha(alpha):
    muAlpha = MuAlpha(alpha)
    lambdaMuAlpha = LambdaBeta(muAlpha)
    gAlpha = (1 - lambdaMuAlpha) * muAlpha
    for i in range(muAlpha):
        pi = p[i]
        gAlpha += pi * (i + 1)
    return gAlpha
def GAdvAlpha(alpha):
    muAlpha = MuAlpha(alpha)
    lambdaMuAlpha = LambdaBeta(muAlpha)
    return math.log(2 * GAlpha(alpha) / lambdaMuAlpha - 1, 2) + math.log(1 / (2 - lambdaMuAlpha), 2)
def GAdv():
    g = 0
    for i in range(len(p)):
        g = g + p[i] * (i + 1)
    return math.log(2 * g - 1, 2)
def _metrics_dataset(dataset):
    global p
    p = getP("result/frequency/{}.txt".format(dataset))
    metrics = {"r1": Hinf(), "r2": LambdaAdvBeta(3), "r3": LambdaAdvBeta(6), "r4": LambdaAdvBeta(30),
               "r5": LambdaAdvBeta(60), "r6": LambdaAdvBeta(100), "r7": LambdaAdvBeta(1000), "r8": GAdvAlpha(0.1),
               "r9": GAdvAlpha(0.2), "r10": GAdvAlpha(0.3), "r11": GAdvAlpha(0.4), "r12": GAdvAlpha(0.5),
               "r13": GAdvAlpha(0.6), "r14": H1(), "r15": GAdv()}
    return metrics
def metrics_datasets():
    dataset2metrics = {dataset: _metrics_dataset(dataset) for dataset in DATASETS}
    save_pickle(metrics_datasets_model.model_path, dataset2metrics)
    return
def _metrics_category(category):
    datasets = CATEGORY2DATASETS[category]
    metrics = {"r{}".format(i): 0 for i in range(1, 16)}
    for l in range(1, 16):
        p = [metrics_datasets_model.get()[dataset]["r{}".format(l)] for dataset in datasets]
        q = [DATASET2SIZE[dataset] for dataset in datasets]
        metrics["r{}".format(l)] = weighted_mean(p, q)
    return metrics
def metrics_categories():
    category2metrics = {category: _metrics_category(category) for category in CATEGORIES}
    save_pickle(metrics_categories_model.model_path, category2metrics)
    return

if __name__ == "__main__":
    # metrics_datasets()
    # metrics_categories()
    print(metrics_categories_model.get())


