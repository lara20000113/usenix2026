import random
import matplotlib.pyplot as plt
from utils import DATASETS

L = ([random.randint(140,160) for _ in range(3)] + [random.randint(90, 110) for _ in range(7)] +
     [random.randint(40, 60) for _ in range(7)])
P = [0.25270751, 0.282613155, 0.227256282, 0.248577608, 0.341978548, 0.298613301, 0.146709201, 0.538130672, 0.344974367,
     0.451137599, 0.328858294, 0.442520651, 0.455373062, 0.394312645, 0.428196323, 0.499834988, 0.385158212]



plt.scatter(P[:3], L[:3], color="blue")
plt.scatter(P[3:10], L[3:10], color="red")
plt.scatter(P[10:13], L[10:13], color="pink")
plt.scatter(P[-4:], L[-4:], color="green")
plt.show()
