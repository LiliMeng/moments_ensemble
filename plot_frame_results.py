import numpy as np
import matplotlib.pyplot as plt

# evenly sampled time at 200ms intervals
top1 = [24.959, 24.917, 25.027, 25.118, 24.950, 24.971, 25.044, 25.012, 25.115, 24.947, 24.938, 24.929, 24.847, 24.714, 24.661]
top5 = [48.776, 48.678, 48.923, 48.855, 49.015, 49.068, 49.056, 48.944, 48.897, 48.906, 48.714, 48.714, 48.673, 48.504, 48.475]
l = [1, 2, 3, 4, 5, 6, 7, 8, 9] #,10, 11, 12, 13, 14, 15]

top1 = [24.858, 25.050, 25.183, 24.897, 24.885, 25.065, 24.799, 24.779, 24.702] 
top5 = [48.968, 48.965, 48.982, 48.876, 48.814, 48.879, 48.560, 48.555, 48.431]

# plt.plot(l, top5, 'g--d',)# l, top5, 'g--')

# plt.xlabel('Frame')
# plt.ylabel('Top5 Accuracy')
# #plt.title('Validation Accuracy Vs Frame')
# # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
# plt.axis([1, 9, 48.4, 49.1])
# #plt.axis([1, 9, 24.6, 25.2])
# plt.grid(True)
# plt.show()
# plt.show()

diff1=max(top1)-min(top1)

diff5=max(top5)-min(top5)

print("diff1: ", diff1)
print("diff5: ", diff5)