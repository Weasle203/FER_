import os
from matplotlib.pyplot import imread
import pandas as pd
import numpy as np



output = []
imgarray = []
def callme(imgd,imglist,df):
	for i in imglist:
	    for j in df:
	        print(j[0])
	        if j[0] == i:
	            output.append(j[1])
	            imgarray.append(imread(os.path.join(imgd,i)))

path = os.getcwd()
for i in os.listdir(path):
	if os.path.isdir(i):
		if i != 'AamiarKhan':
		    for j in os.listdir(os.path.join(path,i)):
		    		if j != '.DS_Store':
			    		try:
				    		imgd = os.path.join(path,i,j,'images')
				    		imglist = os.listdir(imgd)
				    		text = os.path.join(path,i,j,str(j)+'.txt')
				    		df = pd.read_table(text,header = None,usecols = [2,11]).values
				    		callme(imgd,imglist,df)
				    		del df
				    		del imglist
				    		del imgd
				    		del text
				    	except :
				    		continue

		    		
np.savez('data.npz',data = np.array(imgarray),kind = np.array(output))

       

