x = [1,2,3]
b = [0] * len(x)
for i in range(len(x)):
	b[i] = 5 * x[i]
	print(b[i])