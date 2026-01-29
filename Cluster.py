import numpy as np
from sklearn.cluster import KMeans

def cluster(inFileList):
    maxtrix = [

    ]
    X = np.array(matrix)
    kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
    # labelsDic = {i: [] for i in kmeans.labels_}
    # for point, label in zip(points, kmeans.labels_):
    #     labelsDic[label].append(point)
    # for label in labelsDic:
    #     Reuse.subFig([[i for i in range(1, 31)] for j in range(len(labelsDic[label]))], labelsDic[label],
    #                  [None for i in range(len(labelsDic[label]))], ['blue' for i in range(len(labelsDic[label]))], 1,
    #                  30, 0, 1, [i for i in range(1, 31)], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    #                  'Length', 'Proportion', 'Cluster_4_Label_'+str(label)+'.png')
    return kmeans.labels_

if __name__ == '__main__':
    fileList = ['BTC', 'ClixSense', 'Neopets', 'LiveAuctioneers', 'LinkedIn', 'Twitter', 'Wishbone', 'Badoo', 'Fling',
                'Mate1', 'Rockyou', 'Gmail', 'Hotmail', 'Rootkit', 'Xato', 'Yahoo', 'Gawker', 'YouPorn', 'DatPiff']
    print(cluster(fileList))