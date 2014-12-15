import matplotlib.pyplot as plt
import numpy as np
import pickle

folder="results/"
pBetter=[]
pVote=[]
pWeigh=[]
for fold in range(5):
	with open(folder+'u-100k-fold-50-d'+str(fold)+'-top10-results.out', 'rb') as handle:
		out = pickle.load(handle)
	pBetter.append(out[-3][1])
	pVote.append(out[-2][1])
	pWeigh.append(out[-1][1])


x=np.array([1,2,3,4,5])

y = [4, 9, 2,5,6]
z=[1,2,3,5,7]
k=[11,12,13,5,9]

ax = plt.subplot(111)
w = 0.3
ax.bar(x-w, pBetter,width=w,color=(0.1,0.1,0.2),align='center')
ax.bar(x, pVote,width=w,color=(0.2,0.2,0.4),align='center')
ax.bar(x+w, pWeigh,width=w,color=(0.4,0.4,0.6),align='center')
ax.autoscale(tight=True)

plt.show()
