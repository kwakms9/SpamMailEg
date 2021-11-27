import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import SimpleRNN, Embedding, Dense
from keras.models import Sequential

data = pd.read_csv(r'spam.csv',encoding='latin1')
del data['Unnamed: 2']
del data['Unnamed: 3']
del data['Unnamed: 4']
data['v1'] = data['v1'].replace(['ham','spam'],[0,1])
data_X = data['v2']
data_Y = data['v1']

print("개수: ",len(data_X),len(data_Y))

tokenizer =Tokenizer()
tokenizer.fit_on_texts(data_X)  #토큰화
sequences = tokenizer.texts_to_sequences(data_X)
# 앞에서 5개 출력 print(sequences[:5])
word_to_index = tokenizer.word_index

data_X = sequences

max_len = max(len(i) for i in data_X)
print(max_len )
data = pad_sequences(data_X,maxlen=max_len)    #padding

n_of_train = int(len(data_X)*0.8)   #80% 훈련
n_of_test = int(len(data_X) - n_of_train)

test_X = data[n_of_train:] #X_data 데이터 중에서 뒤의 1115개의 데이터만 저장
test_Y = np.array(data_Y[n_of_train:]) #y_data 데이터 중에서 뒤의 1115개의 데이터만 저장
train_X = data[:n_of_train] #X_data 데이터 중에서 앞의 4457개의 데이터만 저장
train_Y = np.array(data_Y[:n_of_train]) #y_data 데이터 중에서 앞의 4457개의 데이터만 저장

#for m,i in enumerate(data_X):
 #   print(m,len(i),i)
vocab_size = len(word_to_index)
model = Sequential()
model.add(Embedding(vocab_size, 32)) # 임베딩 벡터의 차원은 32
model.add(SimpleRNN(32)) # RNN 셀의 hidden_size는 32
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(train_X, train_Y, epochs=4, batch_size=32, validation_split=0.2)    #epochs=4 4번 반복 batch size 한번에 처리량, split 20%를 검증데이터로 사용