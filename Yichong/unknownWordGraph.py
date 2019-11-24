import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ['Total', 'Train', 'Test']
known = [5966942, 3989020, 1943395]
unknown = [64188, 56680, 44123]

x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, known, width, label='Known Data')
rects2 = ax.bar(x + width/2, unknown, width, label='Unknown Data')

ax.set_ylabel('No. of Data')
ax.set_title('Unknown Data Comparison')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()