# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd

np.random.seed(123)

solvers.options['show_progress']=False

n_assests=20
n_obs=106

df = pd.read_csv('Closing_Prices.csv')
# print(df.T)
df_transpose = df.T
# return_vec=np.random.randn(n_assests,n_obs)
return_vec = df_transpose.loc['KOTAKBANK.NS':,:]
return_vec1 = return_vec.to_numpy(np.float64)
# return_vec1.dtype = np.float64

def rand_weights(n):
	k = np.random.rand(n)
	return k / sum(k)

# print(return_vec1)
# print(return_vec1.dtype)
# print(return_vec1.shape[0],return_vec1.shape[1])
# print(np.cov(return_vec1))
# plt.plot(return_vec.T, alpha=.4)
# plt.xlabel('time')
# plt.ylabel('returns')
# plt.show()
	
# print(rand_weights(n_assests))
# print(rand_weights(n_assests))

def random_portfolios(returns):
	p = np.asmatrix(np.mean(returns,axis=1))
	w = np.asmatrix(rand_weights(returns.shape[0]))
	C = np.asmatrix(np.cov(returns))
	mu = w * p.T
	sigma = np.sqrt(w * C * w.T)
# 	print(sigma)
# 	if sigma > 2:
# 		return random_portfolios(returns)
	return mu,sigma
	
n_portfolios = 1000
means, stds = np.column_stack([random_portfolios(return_vec1)for _ in range(n_portfolios)])

plt.plot(stds,means,'o',markersize=5)
plt.xlabel('std')
plt.ylabel('mean')
plt.show()

assets = list(return_vec.index)
print(assets)
# print(return_vec.loc[assets[0],:].max())
# maxi,mini = [(return_vec.loc[i,:].max(),return_vec.loc[i,:].min())for i in assets]
RSV = []
for row in assets:
	Li = return_vec.loc[row,:].min()
	Hi = return_vec.loc[row,:].max()
	diff = Hi - Li
	val = []
	for j in return_vec.loc[row,:]:
		r = ((j - Li)/diff)*100
		val.append(r)
	RSV.append(val)
def kdindex(asset):
	print("For asset:",assets[asset])
	K = []
	D = []
	K.append(50)
	D.append(50)
	for i in range(1,106):
		tmp = RSV[asset][i]*(1/3) + K[i-1]*(2/3)
		K.append(tmp)
		tmp2 = tmp*(1/3)+ D[i-1]*(2/3)
		D.append(tmp2)
		if i == 1:
			continue
		if K[i-1]<=D[i-1] and K[i]>D[i]:
			print("BUY at day ",i)
		if K[i-1]>=D[i-1] and K[i]<D[i]:
			print("SELL at day ", i)
# kdindex(17)	
for i in range(0,20):
	kdindex(i)
# 	maxi.append()
# 	mini.append(return_vec.loc[i,:].min())
# minmax = list(zip(mini,maxi))

# print(return_vec.loc['KOTAKBANK.NS',:].max())