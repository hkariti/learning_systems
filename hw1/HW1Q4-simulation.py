import random

n = 10000
N = 300
k = 5
u = []
u_unbiased = []

for i in xrange(n):
	x = [0 for i in xrange(k)]
	for j in xrange(k):
		x_j = random.randint(1, N)
		while x_j in x:
			x_j = random.randint(1, N)
		x[j] = x_j
	u.append(max(x))
	u_unbiased.append(max(x)*(k+1)/k)

print(str.format("Biased: {0}", sum(u)/len(u)))
print(str.format("UnBiased: {0}", sum(u_unbiased)/len(u_unbiased)))