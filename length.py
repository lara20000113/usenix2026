import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from utils import (DATASETS, DATASET2SIZE,
                   MAX_LENGTH, MIN_LENGTH,
                   CATEGORY2DATASETS, CATEGORIES,
                   save_pickle, load_pickle,
                   weighted_mean)
from read_file import read_file

# 自定义类
class model_manage:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
    def get(self):
        if not self.model:
            self.model = load_pickle(self.model_path)
        return self.model
length_distribution_datasets_model = model_manage("result/Length/length_distribution_datasets.pkl")
length_distribution_categories_model = model_manage("result/Length/length_distribution_categories.pkl")
length_sample_categories_model = model_manage("result/Length/length_sample_categories.pkl")

# 数据统计
def _length_distribution_dataset(dataset):
    read_file_class = read_file()
    g = read_file_class.read("data/{}.txt".format(dataset))
    length2frequency = {l: 0 for l in range(MIN_LENGTH, MAX_LENGTH + 1)}
    while 1:
        try:
            pw = next(g)
            length2frequency[len(pw)] += 1
        except StopIteration:
            break
    return {l: length2frequency[l]/DATASET2SIZE[dataset] for l in range(MIN_LENGTH, MAX_LENGTH + 1)}
def length_distribution_datasets():
    dataset2length_distribution = {dataset: _length_distribution_dataset(dataset) for dataset in DATASETS}
    save_pickle(length_distribution_datasets_model.model_path, dataset2length_distribution)
    return
def _length_distribution_category(category):
    datasets = CATEGORY2DATASETS[category]
    length2proportion = {l: 0 for l in range(MIN_LENGTH, MAX_LENGTH + 1)}
    for l in range(MIN_LENGTH, MAX_LENGTH + 1):
        p = [length_distribution_datasets_model.get()[dataset][l] for dataset in datasets]
        q = [DATASET2SIZE[dataset] for dataset in datasets]
        length2proportion[l] = weighted_mean(p, q)
    return length2proportion
def length_distribution_categories():
    category2length_distribution = {category: _length_distribution_category(category) for category in CATEGORIES}
    save_pickle(length_distribution_categories_model.model_path, category2length_distribution)
    return
def _length_sample_category(category):
    samples = []
    datasets = CATEGORY2DATASETS[category]
    read_file_class = read_file()
    for dataset in datasets:
        g = read_file_class.read("data/{}.txt".format(dataset))
        while 1:
            try:
                pw = next(g)
                samples.append(len(pw))
            except StopIteration:
                break
    return samples
def length_sample_categories():
    category2length_sample = {category: _length_sample_category(category) for category in CATEGORIES}
    save_pickle(length_sample_categories_model.model_path, category2length_sample)
    return
def statistics():
    category2length_sample = length_sample_categories_model.get()
    data = []
    for category in CATEGORIES:
        for length in category2length_sample[category]:
            data.append({'Category': category, 'Length': length})
    df = pd.DataFrame(data)
    stats_list = []
    for category in CATEGORIES:
        data_series = df[df['Category'] == category]['Length']
        stats = {
            'Category': category,
            'Count': len(data_series),
            'Mean': np.mean(data_series),
            'Median': np.median(data_series),
            'Min': np.min(data_series),
            'Max': np.max(data_series),
            'Q1': np.percentile(data_series, 25),
            'Q3': np.percentile(data_series, 75),
            'IQR': np.percentile(data_series, 75) - np.percentile(data_series, 25)
        }
        lower_fence = stats['Q1'] - 1.5 * stats['IQR']
        upper_fence = stats['Q3'] + 1.5 * stats['IQR']
        stats['Lower_Fence'] = lower_fence
        stats['Upper_Fence'] = upper_fence
        stats_list.append(stats)
        # 创建DataFrame
    stats_df = pd.DataFrame(stats_list)
    print("统计结果DataFrame:")
    print("=" * 60)
    print(stats_df.round(2).to_string(index=False))
    print("=" * 60)
    return

# 绘图
def draw_violin():
    category2length_sample = length_sample_categories_model.get()
    data = []
    for category in CATEGORIES:
        for length in category2length_sample[category]:
            data.append({'Category': category, 'Length': length})
    df = pd.DataFrame(data)
    plt.figure(constrained_layout=True, figsize=(6, 3))
    bp = plt.boxplot([df[df['Category'] == category]['Length'].values for category in CATEGORIES],
                     labels=CATEGORIES,
                     patch_artist=True,
                     showmeans=True,
                     meanline=True,
                     showfliers=False, # 不标记出异常值
                     boxprops=dict(facecolor='white', edgecolor='black', linewidth=1),
                     medianprops=dict(color='black', linewidth=1), # 设置中位线为黑色
                     meanprops=dict(color='red', linewidth=1, linestyle='-')) # 设置均值线为红色（与图片一致）
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax = plt.gca()
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(True)
    plt.ylim(1, 15)
    plt.yticks([2,4,6,8,10,12,14], ["2","4","6","8","10","12","14"], fontsize=13)
    plt.xticks([1, 2, 3, 4, 5], ["Financial", "Social", "Email", "Forum", "Content"], fontsize=13)
    plt.tight_layout()
    plt.show()
    return



if __name__ == "__main__":
    length_distribution_datasets()
    length_distribution_categories()
    length_sample_categories()
    statistics()
    draw_violin()

