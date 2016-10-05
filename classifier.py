
# coding: utf-8

# # Submission of [shopping-category-classifier](https://github.com/im9uri/shopping-category-classifier)

# ## Read data

# In[1]:

import pandas as pd


# In[2]:

train_df = pd.read_pickle("../data/soma_goods_train.df")


# In[3]:

train_df.shape


# In[4]:

train_df.head(2)


#  * cate1 는 대분류, cate2 는 중분류, cate3 는 소분류 
#  * 총 10000개의 학습 데이터
#  * 위 id 에 해당되는 이미지 데이터 다운받기 https://www.dropbox.com/s/q0qmx3qlc6gfumj/soma_train.tar.gz

# In[5]:

test_df = pd.read_pickle("../data/test.df")


# In[6]:

test_df.shape


# In[7]:

test_df.tail(2)


# ## Feature Engineering
#  * 최대한 다양한 방식으로 name의 조합을 만들어서, dataframe의 column으로 넣어준다

# #### 1. 한글 형태소 [KoNLPy](http://konlpy.org/)로 형태소 분리

# In[8]:

from konlpy.tag import Kkma
from konlpy.utils import pprint


# In[9]:

def remove_big_words(s):
    #'ㅁㅁㅁㅁ...'가 있으면 뺀다 (KoNLPy의 버그로 글자가 너무 길면 OOM으로 죽으므로, 일단은 가볍게 예외처리)
    ns = ''
    for i in s.split():
        if(u'ㅁㅁㅁㅁ' not in i):
            ns += i
    
    return ns

def get_korean_nouns(s):
    s = remove_big_words(s)
    return ' '.join(kkma.nouns(s))

kkma = Kkma()
train_df['name_korean_nouns'] = train_df['name'].map(get_korean_nouns)
test_df['name_korean_nouns'] = test_df['name'].map(get_korean_nouns)
train_df.head(2)


# #### 2. 숫자랑 알파벳만 분리한다

# In[10]:

def get_is_alnum(s):
    alnum = ''
    for c in str(s):
        if c.isalnum():
            alnum += c
        else:
            alnum += ' '
    return alnum
            
train_df['name_alnum'] = train_df['name'].map(get_is_alnum)
test_df['name_alnum'] = test_df['name'].map(get_is_alnum)
train_df.head(2)


# #### 3. 원래 name 데이터와 위에서 만든 데이터를 더해서 하나의 column으로 만든다

# In[11]:

train_df['name_total'] = train_df['name'] + ' ' + train_df['name_korean_nouns'] + ' ' + train_df['name_alnum']
test_df['name_total'] = test_df['name'] + ' ' + test_df['name_korean_nouns'] + ' ' + test_df['name_alnum']
train_df.head(2)


# #### 4. [Gensim의 word2vec](http://rare-technologies.com/word2vec-tutorial/)을 사용해서 단어들을 neural network의 output으로 뽑아낸다

# In[12]:

#만든 단어의 조합을 한 list에 넣는다
sentences = []
for s in train_df['name_total'].tolist():
    s = s.lower()
    sentences.append(s.split())
    
for s in test_df['name_total'].tolist():
    s = s.lower()
    sentences.append(s.split()) 

    
#Word2Vec model을 만든다
from gensim.models import Word2Vec

#parameter는 적당히 대입한 값으로, tuning 여지가 있다
size = 1000
min_count = 1
model = Word2Vec(sentences, size=size, min_count=1)


# #### 4.1 name_total 안에 있는 각각 단어의 output을 더해서 matrix에 넣는다

# In[13]:

import numpy as np

word_matrix = np.zeros(shape=(train_df.shape[0], size))
index = 0

for s in train_df['name_total'].tolist():
    s = s.lower()
    s_matrix = np.zeros(shape=(size,))
    for w in s.split():
        try:
            s_matrix += model[w]
            break
        except KeyError:
            print('KeyError ' + w)
            
    word_matrix[index] = s_matrix
    index += 1
    
word_matrix.shape


# In[14]:

word_matrix_test = np.zeros(shape=(test_df.shape[0], size))
index = 0

for s in test_df['name_total'].tolist():
    s = s.lower()
    s_matrix = np.zeros(shape=(size,))
    for w in s.split():
        try:
            s_matrix += model[w]
            break
        except KeyError:
            print('KeyError ' + w)
            
    word_matrix_test[index] = s_matrix
    index += 1
    
word_matrix_test.shape


# #### 4.2 다차원의 output을 PCA를 이용해서 차원을 낮춰준다

# In[15]:

from sklearn.decomposition import PCA

#parameter는 적당히 대입한 값으로, tuning 여지가 있다
size = 20
pca = PCA(n_components=size, whiten=True)
pca.fit(word_matrix)

pca.explained_variance_ratio_


# In[16]:

word_matrix = pca.transform(word_matrix)
word_matrix_test = pca.transform(word_matrix_test)

print(word_matrix.shape, word_matrix_test.shape)


# #### 4.3 PCA로 transform한 data를 column으로 넣어준다

# In[17]:

columns = ['word2vec_' + str(i) for i in range(0,size)]
train_word_df = pd.DataFrame(word_matrix, columns=columns, index=train_df.index)
train_df = pd.concat([train_df, train_word_df], axis=1)

test_word_df = pd.DataFrame(word_matrix_test, columns=columns, index=test_df.index)
test_df = pd.concat([test_df, test_word_df], axis=1)

train_df.head(2)


# #### 5. [VGG19 model](https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d)을 이용해서 이미지를 다른 output으로 변환시킨다

# #### 5.1 Define Model

# In[18]:

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
import cv2

def VGG_19(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


# In[19]:

model = VGG_19('../model/vgg19_weights.h5')


# In[20]:

#train된 model은 1000개의 output을 뱉지만, 이 경우에는 그럴 필요가 없으므로 마지막 2개의 layer를 뺀다
model.layers.pop()
model.layers.pop()

#현재 keras 버전(1.0.8)에서는 pop()으로는 layer가 없어지지 않으므로, 밑에 코드를 넣는다
#https://github.com/fchollet/keras/issues/2371#issuecomment-211120172
model.outputs = [model.layers[-1].output]
model.layers[-1].outbound_nodes = []


# #### 5.2 이미지의 output을 저장하고, 이미지가 없는 경우 평균값을 넣어준다

# In[21]:

start = 0
no_img_list = []
fc_matrix = np.zeros(shape=(10000, 4096))

for i in train_df.index.values:
    f = '../data/soma_train/' + str(i) + '.jpg'
    f = cv2.imread(f)
    if(f is None):
        no_img_list.append(start)
        start += 1
        continue
    
    #resize image to fit input
    im = cv2.resize(f, (224, 224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    
    #add to array
    fc_matrix[start] = model.predict_proba(im, verbose=False)
    
    start += 1
    
print(str(len(no_img_list)) + " data with no images")

# no_img_list인 index에 평균값을 넣어준다
avg = np.average(fc_matrix, axis=0)
for i in no_img_list:
    fc_matrix[i] = avg
    
print(fc_matrix.shape)


# In[22]:

start = 0
no_img_list = []
fc_matrix_test = np.zeros(shape=(4807, 4096))

for i in test_df.index.values:
    f = '../data/soma_test/' + str(i) + '.jpg'
    f = cv2.imread(f)
    if(f is None):
        no_img_list.append(start)
        start += 1
        continue
    
    #resize image to fit input
    im = cv2.resize(f, (224, 224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    
    #add to array
    fc_matrix_test[start] = model.predict_proba(im, verbose=False)
    
    start += 1
    
print(str(len(no_img_list)) + " data with no images")

# no_img_list인 index에 평균값을 넣어준다
avg = np.average(fc_matrix_test, axis=0)
for i in no_img_list:
    fc_matrix_test[i] = avg
    
print(fc_matrix_test.shape)


# #### 5.3 다차원의 output을 PCA를 이용해서 차원을 낮춰준다

# In[23]:

from sklearn.decomposition import PCA

#parameter는 적당히 대입한 값으로, tuning 여지가 있다
size = 20
pca = PCA(n_components=size, whiten=True)
pca.fit(fc_matrix)

pca.explained_variance_ratio_


# In[24]:

fc_matrix = pca.transform(fc_matrix)
fc_matrix_test = pca.transform(fc_matrix_test)
print(fc_matrix.shape, fc_matrix_test.shape)


# #### 5.4 PCA로 transform한 data를 column으로 넣어준다

# In[25]:

columns = ['img_' + str(i) for i in range(0,size)]
train_img_df = pd.DataFrame(fc_matrix, columns=columns, index=train_df.index)
train_df = pd.concat([train_df, train_img_df], axis=1)

test_img_df = pd.DataFrame(fc_matrix_test, columns=columns, index=test_df.index)
test_df = pd.concat([test_df, test_img_df], axis=1)

train_df.head(2)


# #### 현재까지 output을 파일로 저장한다

# In[26]:

import pickle

with open('../data/train_feat_eng.df', 'wb') as handle:
  pickle.dump(train_df, handle)

with open('../data/test_feat_eng.df', 'wb') as handle:
  pickle.dump(test_df, handle)


# #### 6. Model에 넣을 수 있는 데이터로 바꾼다

# #### 6.1 단어를 n-gram으로 조합을 만들어서 숫자의 형태로 저장하기 위해 TfidfVectorizer를 사용한다.
#  * CountVectorizer 는 일반 text 를 이에 해당되는 숫자 id 와, 빈도수 형태의 데이터로 변환 해주는 역할을 해준다.
#  * 이 역할을 하기 위해서 모든 단어들에 대해서 id 를 먼저 할당한다.
#  * 그리고 나서, 학습 데이터에서 해당 단어들과, 그것의 빈도수로 데이터를 변환 해준다. (보통 이런 과정을 통해서 우리가 이해하는 형태를 컴퓨터가 이해할 수 있는 형태로 변환을 해준다고 보면 된다)
#  * 예를 들어서 '베네통키즈 키즈 러블리 키즈' 라는 상품명이 있고, 각 단어의 id 가 , 베네통키즈 - 1, 키즈 - 2, 러블리 -3 이라고 한다면 이 상품명은 (1,1), (2,2), (3,1) 형태로 변환을 해준다. (첫번째 단어 id, 두번째 빈도수)
#  * TfidfVectorizer는 CountVectorizer에 tf-idf로 한번 transform 해주는 model이다.

# In[28]:

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 6), strip_accents='unicode')
x_list = vectorizer.fit_transform(train_df['name_total'].tolist())
print(x_list.shape)


# #### 6.2 word2vec, img column을 x_list에 더한다

# In[32]:

from scipy import sparse
from scipy.sparse import hstack

#add word2vec columns
word_columns = ['word2vec_' + str(i) for i in range(0,20)]
word_matrix = sparse.csr_matrix(train_df[word_columns].values)
x_list = hstack([x_list, word_matrix])

#add image columns
img_columns = ['img_' + str(i) for i in range(0,20)]
img_matrix = sparse.csr_matrix(train_df[img_columns].values)

x_list = hstack([x_list, img_matrix])
x_list.shape


# #### 6.3 3개의 카테고리를 제출 형태의 shape로 만든다

# In[33]:

y_list = []
for each in train_df.iterrows():
    s = ';'.join([each[1]['cate1'], each[1]['cate2'], each[1]['cate3']])
    y_list.append(s)


# ## Train Model
#  * 심플하게 Support Vector Machine을 사용한다.

# In[34]:

from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score


# #### 1. GridSearch로 제일 적합한 C값을 찾는다

# In[38]:

svc_param = {'C':np.logspace(-1, 1.5, 30)}
print(svc_param['C'])


# In[ ]:

gs_svc = GridSearchCV(LinearSVC(),svc_param,cv=4,n_jobs=-1, verbose=1)
gs_svc.fit(x_list, y_list)
print(gs_svc.best_params_, gs_svc.best_score_)


# #### 2. Cross validation 점수를 확인한다

# In[ ]:

svc_clf = LinearSVC(C=gs_svc.best_params_['C'])
svc_score = cross_val_score(svc_clf, x_list, y_list, cv=5, n_jobs=-1).mean()
print("LinearSVC = {0:.6f}".format(svc_score))


# #### 3. Predict test data

# #### 3.1 Get test_list

# In[ ]:

test_list = vectorizer.transform(test_df['name_total'].tolist())

#add word2vec columns
word_matrix = sparse.csr_matrix(test_df[word_columns].values)
test_list = hstack([test_list, word_matrix])

#add image columns
img_matrix = sparse.csr_matrix(test_df[img_columns].values)
test_list = hstack([test_list, img_matrix])

print(test_list.shape)


# #### 3.2 Get prediction and save to file

# In[ ]:

svc_clf.fit(x_list, y_list)
pred = svc_clf.predict(test_list)

test_df['pred'] = pred
pred_df = pd.Series(test_df.pred.values,index=test_df.name)
pred_df.head(2)


# In[ ]:

with open('../submission/pred.df', 'wb') as handle:
  pickle.dump(pred_df, handle)


# ## Setup server, and check final score
#  * http://somaeval.hoot.co.kr:8880/eval?url=http://52.41.52.48:8887 (check only two categories)
#  * http://somaeval.hoot.co.kr:8880/eval?url=http://52.41.52.48:8887&mode=all&name=임규리 (full test)
#  * http://somaeval.hoot.co.kr:8869/score (leaderboard)

# In[ ]:

d = pred_df.to_dict()

%%capture
from bottle import route, run, template,request,get, post
import re
import  time
from threading import  Condition
_CONDITION = Condition()
@route('/classify')

def classify():
    img = request.query.img
    name = request.query.name
    pred = d[name]
    
    return {'cate':pred}

run(host='0.0.0.0', port=8887)

