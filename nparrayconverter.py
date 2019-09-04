import pandas as pd
import numpy as np

df = pd.read_csv('inventory/fer2013.csv')

l=[]
for i in df.pixels:
	l.append(list(map(int,i.split(' '))))
print(l)

np.savez('inventory/ferarray.npz', data = np.array(l),cls=df.emotion)