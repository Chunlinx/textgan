from nltk.tokenize import sent_tokenize, word_tokenize
import re
import numpy as np

data = open("cerpen.txt", encoding="utf8")
data = data.read()
#data = re.sub("(.*?)","",data)
#data = re.sub("\n"," ",data)
token = word_tokenize(data)
data = sent_tokenize(data)

# create mapping of unique chars to integers
token = sorted(list(set(token)))
dictionary = dict((c, i) for i, c in enumerate(token, 1))
dictionary[''] = 0

word2int = dictionary
int2word = dict((c, i) for i, c in dictionary.items())

dataX = []
for i in range(0, len(data)):
    dataseq = word_tokenize(data[i])
    dataX.append([dictionary[char] for char in dataseq])

for i in range(0, len(data)):
    data[i] = word_tokenize(data[i])

label = []
for i in range(0, len(data)):
    if data[i][len(data[i])-1] == '.':
        label.append('1')
    else:
        label.append('0')


maxlen = 0
for i in range(len(dataX)):
	if maxlen < len(dataX[i]):
		maxlen = len(dataX[i])
        
print(maxlen)
for i in range(len(data)):
    if len(data[i]) == maxlen-1:
        print(data[i])


dataNP = np.zeros((len(dataX),maxlen,1), dtype=np.int)
for i in range(0,len(dataX)):
    for j in range(0,len(dataX[i])):
        dataNP[i][j] = dataX[i][j]

x = [None] * 6
x[0] = dataNP
x[1] = dataX                  
x[2] = data
x[3] = label
x[4], x[5]    = word2int, int2word

import pickle
with open('wonderland.p', 'wb') as handle:
	pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)