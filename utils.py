import numpy as np
from scipy import stats
from read_file import read_file_format_Chi_Squared_data
import pickle

DATASETS = ["BTC", "ClixSense", "LiveAuctioneers",
            "LinkedIn", "Twitter", "Wishbone", "Badoo", "Fling",
            "Gmail", "Hotmail",
            "Rootkit", "Xato", "Rockyou",
            "Yahoo", "Gawker", "YouPorn", "DatPiff"]
DATASET2SIZE = {"BTC": 23624127, "ClixSense": 2911665, "LiveAuctioneers": 2222046,
                "LinkedIn": 54639458, "Twitter": 25350096, "Wishbone": 10195862, "Badoo": 25881848, "Fling": 40742503,
                "Gmail": 4899553, "Hotmail": 9704223,
                "Rootkit": 69325, "Xato": 9997772, "Rockyou": 32572338,
                "Yahoo": 5616792, "Gawker": 607753, "YouPorn": 2140079, "DatPiff": 7157711}
CATEGORIES = ["financial", "social", "email", "forum", "content"]
CATEGORY2DATASETS = {
                        "financial": ["BTC", "ClixSense", "LiveAuctioneers"],
                        "social": ["LinkedIn", "Twitter", "Wishbone", "Badoo", "Fling"],
                        "email": ["Gmail", "Hotmail"],
                        "forum": ["Rootkit", "Xato", "Rockyou"],
                        "content": ["Yahoo", "Gawker", "YouPorn", "DatPiff"]
                     }
DATASETS_PII = ["BTC", "ClixSense", "LiveAuctioneers", "LinkedIn", "Yahoo", "DatPiff"]
CATEGORIES_PII = ["financial", "general"]
CATEGORY2DATASETS_PII = {
                        "financial": ["BTC", "ClixSense", "LiveAuctioneers"],
                        "general": ["Yahoo", "LinkedIn", "DatPiff"]
                     }
MIN_LENGTH, MAX_LENGTH = 1, 30

def chi_squared(model):
    items = list(model['BTC'].keys())
    parameters = {item: {category: [0, 0] for category in CATEGORIES} for item in items}
    for item in items:
        for category in CATEGORIES:
            datasets = CATEGORY2DATASETS[category]
            parameters[item][category][0] += sum(
                [model[dataset][item] * DATASET2SIZE[dataset] for dataset in datasets])
            parameters[item][category][1] = sum([DATASET2SIZE[dataset] for dataset in datasets]) - parameters[item][category][0]
        for category in CATEGORIES[1:]:
            observe = np.array([parameters[item][CATEGORIES[0]], parameters[item][category]])
            chi2, p, dof, expected = stats.chi2_contingency(observe)
            print(item, CATEGORIES[0], "vs.", category, chi2, dof, p,
                  parameters[item][CATEGORIES[0]][0] / sum(parameters[item][CATEGORIES[0]]),
                  parameters[item][category][0] / sum(parameters[item][category]))
    return
def save_pickle(file, model):
    with open(file, 'wb') as f:  # 注意：二进制写入模式 'wb'
        pickle.dump(model, f)
def load_pickle(file):
    with open(file, 'rb') as f:  # 注意：二进制读取模式 'rb'
        return pickle.load(f)
def weighted_mean(distribution, weight):
    return sum([d*w for d, w in zip(distribution, weight)])/sum(weight)


