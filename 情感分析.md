
* Kaggle比赛情感分析题目：Bag of Words Meets Bags of Popcorn
* Kaggle比赛地址：https://www.kaggle.com/c/word2vec-nlp-tutorial#description

# 数据特征处理

## 一、数据预处理

### 1、加载工具包


```python
import pandas as pd
import numpy as np
```

### 2、读取并查看数据


```python
root_dir = "data"
# 载入数据集
train = pd.read_csv('%s/%s' % (root_dir, 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=1)
unlabel_train = pd.read_csv('%s/%s' % (root_dir, 'unlabeledTrainData.tsv'), header=0, delimiter="\t", quoting=3)   
test = pd.read_csv('%s/%s' % (root_dir, 'testData.tsv'), header=0, delimiter="\t", quoting=1)

print(train.shape)
print(train.columns.values)
print(unlabel_train.shape)
print(unlabel_train.columns.values)
print(test.shape)
print(test.columns.values)

print(train.head(3))
print(unlabel_train.head(3))
print(test.head(3))
```

    (25000, 3)
    ['id' 'sentiment' 'review']
    (50000, 2)
    ['id' 'review']
    (25000, 2)
    ['id' 'review']
           id  sentiment                                             review
    0  5814_8          1  With all this stuff going down at the moment w...
    1  2381_9          1  \The Classic War of the Worlds\" by Timothy Hi...
    2  7759_3          0  The film starts with a manager (Nicholas Bell)...
              id                                             review
    0   "9999_0"  "Watching Time Chasers, it obvious that it was...
    1  "45057_0"  "I saw this film about 20 years ago and rememb...
    2  "15561_0"  "Minor Spoilers<br /><br />In New York, Joan B...
             id                                             review
    0  12311_10  Naturally in a film who's main themes are of m...
    1    8348_2  This movie is a disaster within a disaster fil...
    2    5828_4  All in all, this is a movie for kids. We saw i...
    

#### 从原始数据中可以看出：
* 1.labeledTrainData数据用于模型训练；unlabeledTrainData数据用于Word2vec提取特征；testData数据用于提交结果预测。
* 2.文本数据来自网络爬虫数据，带有html格式

### 3.去除HTML标签+数字+全部小写


```python
def review_to_wordlist(review):
    '''
    把IMDB的评论转成词序列
    '''
    from bs4 import BeautifulSoup
    # 1.去掉HTML标签，拿到内容
    review_text = BeautifulSoup(review, "html.parser").get_text()
    
    import re
    # 用正则表达式取出符合规范的部分
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    
    # 小写化所有的词，并转成词list
    words_list = review_text.lower().split()
    
    #去除停用词。需要下载nltk库，并且下载stopwords。
    from nltk.corpus import stopwords
    stopwords = set(stopwords.words("english"))
    words = [word for word in words_list if word not in stopwords]
    
    # 返回words
    return words

# 预处理数据
label = train['sentiment']
train_data = []
for i in range(len(train['review'])):
    train_data.append(' '.join(review_to_wordlist(train['review'][i])))
    
unlable_data = []    
for i in range(len(unlabel_train['review'])):
    unlable_data.append(' '.join(review_to_wordlist(unlabel_train['review'][i])))   
    
test_data = []
for i in range(len(test['review'])):
    test_data.append(' '.join(review_to_wordlist(test['review'][i])))

# 预览数据
print(train_data[0], '\n')
print(unlable_data[0], '\n')
print(test_data[0])
```

    stuff going moment mj started listening music watching odd documentary watched wiz watched moonwalker maybe want get certain insight guy thought really cool eighties maybe make mind whether guilty innocent moonwalker part biography part feature film remember going see cinema originally released subtle messages mj feeling towards press also obvious message drugs bad kay visually impressive course michael jackson unless remotely like mj anyway going hate find boring may call mj egotist consenting making movie mj fans would say made fans true really nice actual feature film bit finally starts minutes excluding smooth criminal sequence joe pesci convincing psychopathic powerful drug lord wants mj dead bad beyond mj overheard plans nah joe pesci character ranted wanted people know supplying drugs etc dunno maybe hates mj music lots cool things like mj turning car robot whole speed demon sequence also director must patience saint came filming kiddy bad sequence usually directors hate working one kid let alone whole bunch performing complex dance scene bottom line movie people like mj one level another think people stay away try give wholesome message ironically mj bestest buddy movie girl michael jackson truly one talented people ever grace planet guilty well attention gave subject hmmm well know people different behind closed doors know fact either extremely nice stupid guy one sickest liars hope latter 
    
    watching time chasers obvious made bunch friends maybe sitting around one day film school said hey let pool money together make really bad movie something like ever said still ended making really bad movie dull story bad script lame acting poor cinematography bottom barrel stock music etc corners cut except one would prevented film release life like 
    
    naturally film main themes mortality nostalgia loss innocence perhaps surprising rated highly older viewers younger ones however craftsmanship completeness film anyone enjoy pace steady constant characters full engaging relationships interactions natural showing need floods tears show emotion screams show fear shouting show dispute violence show anger naturally joyce short story lends film ready made structure perfect polished diamond small changes huston makes inclusion poem fit neatly truly masterpiece tact subtlety overwhelming beauty
    

#### 查看数据量


```python
print(len(train_data),len(unlable_data),len(test_data))
```

    25000 50000 25000
    

## 二、特征工程

把文本转换为向量，有几种常见的文本向量处理方法，比如：

* 1.单词计数
* 2.TF-IDF向量
* 3.Word2vec向量

### 1.Count词向量


```python
from sklearn.feature_extraction.text import CountVectorizer

count_vec = CountVectorizer(
    max_features=4000,#过滤掉低频
    analyzer='word', # tokenise by character ngrams
    ngram_range=(1,2),  # 二元n-gram模型
    stop_words = 'english')

# 合并训练和测试集以便进行TFIDF向量化操作
data_all = train_data + test_data
len_train = len(train_data)

count_vec.fit(data_all)
data_all = count_vec.transform(data_all)

# 恢复成训练集和测试集部分
count_train_x = data_all[:len_train]
count_test_x = data_all[len_train:]

print('count处理结束.')

print("train: \n", np.shape(count_train_x[0]))
print("test: \n", np.shape(count_test_x[0]))
```

    count处理结束.
    train: 
     (1, 4000)
    test: 
     (1, 4000)
    


```python
count_train_x.shape
```




    (25000, 4000)




```python
count_test_x.shape
```




    (25000, 4000)



#### Count特征
* count_train_x
* count_test_x

### 2.TF-IDF词向量


```python
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
"""
min_df: 最小支持度为2（词汇出现的最小次数）
max_features: 默认为None，可设为int，对所有关键词的term frequency进行降序排序，只取前max_features个作为关键词集
strip_accents: 将使用ascii或unicode编码在预处理步骤去除raw document中的重音符号
analyzer: 设置返回类型
token_pattern: 表示token的正则表达式，需要设置analyzer == 'word'，默认的正则表达式选择2个及以上的字母或数字作为token，标点符号默认当作token分隔符，而不会被当作token
ngram_range: 词组切分的长度范围
use_idf: 启用逆文档频率重新加权
use_idf：默认为True，权值是tf*idf，如果设为False，将不使用idf，就是只使用tf，相当于CountVectorizer了。
smooth_idf: idf平滑参数，默认为True，idf=ln((文档总数+1)/(包含该词的文档数+1))+1，如果设为False，idf=ln(文档总数/包含该词的文档数)+1
sublinear_tf: 默认为False，如果设为True，则替换tf为1 + log(tf)
stop_words: 设置停用词，设为english将使用内置的英语停用词，设为一个list可自定义停用词，设为None不使用停用词，设为None且max_df∈[0.7, 1.0)将自动根据当前的语料库建立停用词表
"""
tfidf = TFIDF(min_df=2,
           max_features=4000,#过滤掉低频
           strip_accents='unicode',
           analyzer='word',
           token_pattern=r'\w{1,}',
           ngram_range=(1,2),  # 二元n-gram模型
           use_idf=1,
           smooth_idf=1,
           sublinear_tf=1,
           stop_words = 'english') # 去掉英文停用词

# 合并训练和测试集以便进行TFIDF向量化操作
data_all = train_data + test_data
len_train = len(train_data)

tfidf.fit(data_all)
data_all = tfidf.transform(data_all)
# 恢复成训练集和测试集部分
tfidf_train_x = data_all[:len_train]
tfidf_test_x = data_all[len_train:]
print('TF-IDF处理结束.')

print("train: \n", np.shape(tfidf_train_x[0]))
print("test: \n", np.shape(tfidf_test_x[0]))
```

    TF-IDF处理结束.
    train: 
     (1, 4000)
    test: 
     (1, 4000)
    


```python
tfidf_train_x.shape
```




    (25000, 4000)




```python
tfidf_test_x.shape
```




    (25000, 4000)



### 3.Word2vec词向量

* gensim.models.word2vec.Word2Vec 输入数据是字符的list格式，所以需要对数据进行预处理

#### 3.1 输入数据预处理


```python
#预处理训练数据
train_words = []
for i in train_data:
    train_words.append(i.split())
    
#预处理特征数据
unlable_words = []
for i in unlable_data:
    unlable_words.append(i.split())

#预处理测试数据
test_words = []
for i in test_data:
    test_words.append(i.split())

#合并数据
all_words = train_words + unlable_words + test_words

len(all_words)
```




    100000



#### 3.2数据预览


```python
# 预览数据
print(all_words[0])
```

    ['stuff', 'going', 'moment', 'mj', 'started', 'listening', 'music', 'watching', 'odd', 'documentary', 'watched', 'wiz', 'watched', 'moonwalker', 'maybe', 'want', 'get', 'certain', 'insight', 'guy', 'thought', 'really', 'cool', 'eighties', 'maybe', 'make', 'mind', 'whether', 'guilty', 'innocent', 'moonwalker', 'part', 'biography', 'part', 'feature', 'film', 'remember', 'going', 'see', 'cinema', 'originally', 'released', 'subtle', 'messages', 'mj', 'feeling', 'towards', 'press', 'also', 'obvious', 'message', 'drugs', 'bad', 'kay', 'visually', 'impressive', 'course', 'michael', 'jackson', 'unless', 'remotely', 'like', 'mj', 'anyway', 'going', 'hate', 'find', 'boring', 'may', 'call', 'mj', 'egotist', 'consenting', 'making', 'movie', 'mj', 'fans', 'would', 'say', 'made', 'fans', 'true', 'really', 'nice', 'actual', 'feature', 'film', 'bit', 'finally', 'starts', 'minutes', 'excluding', 'smooth', 'criminal', 'sequence', 'joe', 'pesci', 'convincing', 'psychopathic', 'powerful', 'drug', 'lord', 'wants', 'mj', 'dead', 'bad', 'beyond', 'mj', 'overheard', 'plans', 'nah', 'joe', 'pesci', 'character', 'ranted', 'wanted', 'people', 'know', 'supplying', 'drugs', 'etc', 'dunno', 'maybe', 'hates', 'mj', 'music', 'lots', 'cool', 'things', 'like', 'mj', 'turning', 'car', 'robot', 'whole', 'speed', 'demon', 'sequence', 'also', 'director', 'must', 'patience', 'saint', 'came', 'filming', 'kiddy', 'bad', 'sequence', 'usually', 'directors', 'hate', 'working', 'one', 'kid', 'let', 'alone', 'whole', 'bunch', 'performing', 'complex', 'dance', 'scene', 'bottom', 'line', 'movie', 'people', 'like', 'mj', 'one', 'level', 'another', 'think', 'people', 'stay', 'away', 'try', 'give', 'wholesome', 'message', 'ironically', 'mj', 'bestest', 'buddy', 'movie', 'girl', 'michael', 'jackson', 'truly', 'one', 'talented', 'people', 'ever', 'grace', 'planet', 'guilty', 'well', 'attention', 'gave', 'subject', 'hmmm', 'well', 'know', 'people', 'different', 'behind', 'closed', 'doors', 'know', 'fact', 'either', 'extremely', 'nice', 'stupid', 'guy', 'one', 'sickest', 'liars', 'hope', 'latter']
    

#### 3.3 Word2vec模型训练保存


```python
from gensim.models.word2vec import Word2Vec
import os

# 设定词向量训练的参数
size = 100      # Word vector dimensionality
min_count = 3   # Minimum word count
num_workers = 4 # Number of threads to run in parallel
window = 10     # Context window size
model_name = '{}size_{}min_count_{}window.model'.format(size, min_count, window)

wv_model = Word2Vec(all_words, workers=num_workers, size=size, min_count = min_count,window = window)
wv_model.init_sims(replace=True)#模型训练好后，锁定模型

wv_model.save(model_name)#保存模型
```

    C:\ProgramData\Anaconda3\lib\site-packages\gensim\utils.py:1197: UserWarning: detected Windows; aliasing chunkize to chunkize_serial
      warnings.warn("detected Windows; aliasing chunkize to chunkize_serial")
    

#### 3.4 Word2vec模型加载


```python
from gensim.models.word2vec import Word2Vec
wv_model = Word2Vec.load("100size_3min_count_10window.model")
```

#### 3.5 word2vec特征处理

* 此处画风比较奇特:
* 将一个句子对应的词向量求和取平均，做为机器学习的特征，但是效果还不错。


```python
def to_review_vector(review):
    global word_vec
    word_vec = np.zeros((1,100))
    for word in review:
        if word in wv_model:
            word_vec += np.array([wv_model[word]])
    return pd.Series(word_vec.mean(axis = 0))

train_data_features = []

for i in train_words:
    train_data_features.append(to_review_vector(i))

test_data_features = []
for i in test_words:
    test_data_features.append(to_review_vector(i))
```

    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel\__main__.py:5: DeprecationWarning: Call to deprecated `__contains__` (Method will be removed in 4.0.0, use self.wv.__contains__() instead).
    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel\__main__.py:6: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
    

# 机器学习方法

## 一、机器学习建模

### 1. TF-IDF+朴素贝叶斯模型+交叉验证


```python
# 朴素贝叶斯训练

from sklearn.naive_bayes import MultinomialNB as MNB

model_NB = MNB() # (alpha=1.0, class_prior=None, fit_prior=True)
# 为了在预测的时候使用
model_NB.fit(tfidf_train_x, label)

from sklearn.model_selection import cross_val_score
import numpy as np

print("多项式贝叶斯分类器10折交叉验证得分:  \n", cross_val_score(model_NB, tfidf_train_x, label, cv=10, scoring='roc_auc'))
print("\n多项式贝叶斯分类器10折交叉验证得分: ", np.mean(cross_val_score(model_NB, tfidf_train_x, label, cv=10, scoring='roc_auc')))
test_predicted = np.array(model_NB.predict(tfidf_test_x))

print('保存结果...')

submission_df = pd.DataFrame({'id': test['id'].values, 'sentiment': test_predicted})
print(submission_df.head())
submission_df.to_csv('submission_nb.csv', index = False)

print('结束.')
```

    多项式贝叶斯分类器10折交叉验证得分:  
     [0.93355328 0.9274848  0.93499392 0.93026432 0.93225216 0.93138432
     0.93329536 0.93749696 0.92288384 0.9314976 ]
    
    多项式贝叶斯分类器10折交叉验证得分:  0.931510656
    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          1
    3    7186_2          1
    4   12128_7          1
    结束.
    

![image.png](attachment:image.png)

![image.png](attachment:image.png)

### 2. TF-IDF+逻辑回归模型+网格搜索交叉验证


```python
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import GridSearchCV

# 设定grid search的参数
grid_values = {'C': [0.1, 1, 10]}  
# 设定打分为roc_auc
"""
penalty: l1 or l2, 用于指定惩罚中使用的标准。
"""
model_LR = GridSearchCV(LR(penalty='l2', dual=True, random_state=0), grid_values, scoring='roc_auc', cv=5)
model_LR.fit(tfidf_train_x, label)

# 输出结果
print("最好的参数：")
print( model_LR.best_params_)

print("最好的得分：")
print(model_LR.best_score_)


print("网格搜索参数及得分：")
print(model_LR.grid_scores_)

print("网格搜索结果：")
print(model_LR.cv_results_)

model_LR = LR(penalty='l2', dual=True, random_state=0)
model_LR.fit(tfidf_train_x, label)

test_predicted = np.array(model_LR.predict(tfidf_test_x))

print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(10))
submission_df.to_csv('submission_lr.csv',columns = ['id','sentiment'], index = False)
print('结束.')
```

    最好的参数：
    {'C': 1}
    最好的得分：
    0.9507756479999999
    网格搜索参数及得分：
    [mean: 0.93949, std: 0.00324, params: {'C': 0.1}, mean: 0.95078, std: 0.00276, params: {'C': 1}, mean: 0.94398, std: 0.00275, params: {'C': 10}]
    网格搜索结果：
    {'mean_fit_time': array([0.13519988, 0.17780013, 0.53299994]), 'std_fit_time': array([0.01399142, 0.01008764, 0.06999427]), 'mean_score_time': array([0.00260015, 0.00279984, 0.00359998]), 'std_score_time': array([0.00048994, 0.00039988, 0.00174354]), 'param_C': masked_array(data=[0.1, 1, 10],
                 mask=[False, False, False],
           fill_value='?',
                dtype=object), 'params': [{'C': 0.1}, {'C': 1}, {'C': 10}], 'split0_test_score': array([0.93795456, 0.95020352, 0.9436768 ]), 'split1_test_score': array([0.94196192, 0.9542512 , 0.94876688]), 'split2_test_score': array([0.93759728, 0.9471152 , 0.9402704 ]), 'split3_test_score': array([0.9444336 , 0.95359056, 0.94413712]), 'split4_test_score': array([0.93548752, 0.94871776, 0.94303536]), 'mean_test_score': array([0.93948698, 0.95077565, 0.94397731]), 'std_test_score': array([0.00324066, 0.00275551, 0.00274533]), 'rank_test_score': array([3, 1, 2]), 'split0_train_score': array([0.9499337 , 0.97234608, 0.9860402 ]), 'split1_train_score': array([0.94941395, 0.97180812, 0.98526184]), 'split2_train_score': array([0.95103983, 0.97310123, 0.98656722]), 'split3_train_score': array([0.94978255, 0.97232675, 0.98596239]), 'split4_train_score': array([0.95038084, 0.97259464, 0.98615951]), 'mean_train_score': array([0.95011017, 0.97243536, 0.98599823]), 'std_train_score': array([0.0005587 , 0.00041999, 0.0004231 ])}
    

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_search.py:761: DeprecationWarning: The grid_scores_ attribute was deprecated in version 0.18 in favor of the more elaborate cv_results_ attribute. The grid_scores_ attribute will not be available from 0.20
      DeprecationWarning)
    

    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          1
    3    7186_2          1
    4   12128_7          1
    5    2913_8          1
    6    4396_1          0
    7     395_2          0
    8   10616_1          0
    9    9074_9          1
    结束.
    

![image.png](attachment:image.png)

![image.png](attachment:image.png)

### 3. TF-IDF+SVM模型+网格搜索交叉验证

* SVM模型训练太耗时，尤其是使用网格搜索训练
* 参数调优时，使用param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}+5折交叉验证，连续训练时长将近48小时。


```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

'''
线性的SVM只需要，只需要调优正则化参数C
基于RBF核的SVM，需要调优gamma参数和C
'''
param_grid = {'C': [10],'gamma': [1]}

model_SVM = GridSearchCV(SVC(), param_grid, scoring='roc_auc', cv=5)
model_SVM.fit(tfidf_train_x, label)

# 输出结果
print("最好的参数：")
print( model_SVM.best_params_)

print("最好的得分：")
print(model_SVM.best_score_)

print("网格搜索参数及得分：")
print(model_SVM.grid_scores_)

```

    最好的参数：
    {'C': 10, 'gamma': 1}
    最好的得分：
    0.9517343039999999
    网格搜索参数及得分：
    [mean: 0.95173, std: 0.00264, params: {'C': 10, 'gamma': 1}]
    

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_search.py:761: DeprecationWarning: The grid_scores_ attribute was deprecated in version 0.18 in favor of the more elaborate cv_results_ attribute. The grid_scores_ attribute will not be available from 0.20
      DeprecationWarning)
    


```python
from sklearn.externals import joblib
joblib.dump(model_SVM, "model_SVM")
model_SVM = joblib.load("model_SVM")
```


```python
from sklearn.svm import SVC
svm = SVC(kernel='linear',C=10,gamma = 1)
svm.fit(train_x, label)

test_predicted = np.array(svm.predict(test_x))
print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_svm_best.csv',columns = ['id','sentiment'], index = False)
print('结束.')
```

    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          1
    3    7186_2          1
    4   12128_7          1
    5    2913_8          1
    6    4396_1          0
    7     395_2          0
    8   10616_1          0
    9    9074_9          0
    结束.
    

![image.png](attachment:image.png)

### 4. TF-IDF+MLP（多层感知机模型）


```python
#None 维度在这里是一个 batch size 的占位符

import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential

model = Sequential()
#model.add(Dense(3, activation='relu', input_shape=(18,)))

model.add(Dense(10, input_shape=(4000,), activation='relu'))#

model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

y_train_keras = np.array(keras.utils.to_categorical(label, 2))

# Training the model
model.fit(train_x, y_train_keras, epochs=30, batch_size=5000, verbose=0)

```

    C:\ProgramData\Anaconda3\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.
    

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 10)                40010     
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 10)                0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 22        
    =================================================================
    Total params: 40,032
    Trainable params: 40,032
    Non-trainable params: 0
    _________________________________________________________________
    




    <keras.callbacks.History at 0x2d2b5b38>




```python
test_predicted = np.array(model.predict_classes(test_x))
print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_mlp.csv',columns = ['id','sentiment'], index = False)
print('结束.')
```

    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          0
    3    7186_2          0
    4   12128_7          1
    结束.
    

## 二、Word2vec+机器学习建模

### 1.Word2vec+LR


```python
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import cross_val_score

wv_model_LR = LR(penalty='l2', dual=True, random_state=0)
wv_model_LR.fit(train_data_features, label)

scores = cross_val_score(wv_model_LR, train_data_features, label, cv=10, scoring='roc_auc')

print("LR分类器 10折交叉验证得分: \n", scores)
print("LR分类器 10折交叉验证平均得分: \n", np.mean(scores))

test_predicted = np.array(wv_model_LR.predict(test_data_features))
print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_lr_wv.csv',columns = ['id','sentiment'], index = False)
print('结束.')
```

    
    LR分类器 10折交叉验证得分: 
     [0.94150016 0.940256   0.94892224 0.94045184 0.93707328 0.94096768
     0.94216512 0.94608384 0.93529472 0.94407232]
    
    LR分类器 10折交叉验证平均得分: 
     0.94167872
    

    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel\__main__.py:5: DeprecationWarning: Call to deprecated `__contains__` (Method will be removed in 4.0.0, use self.wv.__contains__() instead).
    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel\__main__.py:6: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
    

    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          1
    3    7186_2          0
    4   12128_7          1
    结束.
    

![image.png](attachment:image.png)

### 2.Word2vec+GNB


```python
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.cross_validation import cross_val_score

wv_gnb_model = GNB()
wv_gnb_model.fit(train_data_features, label)


scores = cross_val_score(wv_gnb_model, train_data_features, label, cv=10, scoring='roc_auc')
print("\n高斯贝叶斯分类器 10折交叉验证得分: \n", scores)
print("\n高斯贝叶斯分类器 10折交叉验证平均得分: \n", np.mean(scores))

test_predicted = np.array(wv_gnb_model.predict(test_data_features))
print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_gnb_wv.csv',columns = ['id','sentiment'], index = False)
print('结束.')
```

    
    高斯贝叶斯分类器 10折交叉验证得分: 
     [0.805312   0.78392544 0.79209728 0.8003056  0.80113344 0.78988256
     0.80728544 0.80764544 0.7916688  0.80663168]
    
    高斯贝叶斯分类器 10折交叉验证平均得分: 
     0.798588768
    

    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel\__main__.py:5: DeprecationWarning: Call to deprecated `__contains__` (Method will be removed in 4.0.0, use self.wv.__contains__() instead).
    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel\__main__.py:6: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
    

    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          0
    3    7186_2          0
    4   12128_7          0
    结束.
    

![image.png](attachment:image.png)

### 3.Word2vec+Knn


```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

wv_knn_model = KNeighborsClassifier(n_neighbors=5)
wv_knn_model.fit(train_data_features, label)

scores = cross_val_score(wv_knn_model, train_data_features, label, cv=10, scoring='roc_auc')

print("\nknn算法 10折交叉验证得分: \n", scores)
print("\nknn算法 10折交叉验证平均得分: \n", np.mean(scores))

test_predicted = np.array(wv_knn_model.predict(test_data_features))
print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_knn_wv.csv',columns = ['id','sentiment'], index = False)
print('结束.')
```

    
    knn算法 10折交叉验证得分: 
     [0.89709856 0.89423168 0.90665152 0.8890768  0.89263712 0.89179264
     0.88568    0.90213216 0.89707968 0.88941376]
    
    knn算法 10折交叉验证平均得分: 
     0.894579392
    

    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel\__main__.py:5: DeprecationWarning: Call to deprecated `__contains__` (Method will be removed in 4.0.0, use self.wv.__contains__() instead).
    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel\__main__.py:6: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
    

    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          0
    3    7186_2          0
    4   12128_7          1
    结束.
    

### 4.Word2vec+DTC


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

wv_tree_model = DecisionTreeClassifier()
wv_tree_model.fit(train_data_features, label)

scores = cross_val_score(wv_tree_model, train_data_features, label, cv=5, scoring='roc_auc')

print("决策树 10折交叉验证得分: \n", scores)
print("决策树 10折交叉验证平均得分: \n", np.mean(scores))

test_predicted = np.array(wv_tree_model.predict(test_data_features))

print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_dtc_wv.csv',columns = ['id','sentiment'], index = False)
print('结束.')
```

    
    决策树 10折交叉验证得分: 
     [0.7498 0.7506 0.7572 0.7648 0.736 ]
    
    决策树 10折交叉验证平均得分: 
     0.75168
    

    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel\__main__.py:5: DeprecationWarning: Call to deprecated `__contains__` (Method will be removed in 4.0.0, use self.wv.__contains__() instead).
    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel\__main__.py:6: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
    

    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          0
    3    7186_2          0
    4   12128_7          1
    结束.
    

### 5.Word2vec+xgboost


```python
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import numpy as np
import pandas as pd

xgb_model = XGBClassifier(n_estimators=150, min_samples_leaf=3, max_depth=6)
"""
AttributeError: 'list' object has no attribute 'shape'
list => np.array
"""
wv_xgb_model.fit(pd.DataFrame(train_data_features), label)

scores = cross_val_score(wv_xgb_model, pd.DataFrame(train_data_features), label, cv=10, scoring='roc_auc')

print("XGB 10折交叉验证得分: \n", scores)
print("XGB 10折交叉验证平均得分: \n", np.mean(scores))

test_predicted = np.array(wv_xgb_model.predict(pd.DataFrame(test_data_features)))

print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_xgb_wv.csv',columns = ['id','sentiment'], index = False)
print('结束.')
```

    
    XGB 10折交叉验证得分: 
     [0.94352704 0.94358528 0.95147776 0.94447616 0.9414528  0.94390464
     0.94479296 0.94848128 0.94080992 0.94563072]
    
    XGB 10折交叉验证平均得分: 
     0.944813856
    

    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel\__main__.py:5: DeprecationWarning: Call to deprecated `__contains__` (Method will be removed in 4.0.0, use self.wv.__contains__() instead).
    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel\__main__.py:6: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
    

    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          1
    3    7186_2          0
    4   12128_7          1
    结束.
    

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\preprocessing\label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:
    

### 6.Word2vec+RF


```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

wv_rf_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=0)
wv_rf_model.fit(train_data_features, label)

scores = cross_val_score(wv_rf_model, train_data_features, label, cv=10, scoring='roc_auc')

print("\n随机森林 10折交叉验证得分: \n", scores)
print("\n随机森林 10折交叉验证平均得分: \n", np.mean(scores))

test_predicted = np.array(wv_rf_model.predict(test_data_features))
print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_rfc_wv.csv',columns = ['id','sentiment'], index = False)
print('结束.')
```

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\ensemble\weight_boosting.py:29: DeprecationWarning: numpy.core.umath_tests is an internal NumPy module and should not be imported. It will be removed in a future NumPy release.
      from numpy.core.umath_tests import inner1d
    

    
    随机森林 10折交叉验证得分: 
     [0.9188896  0.9112768  0.92621696 0.9159808  0.91992    0.91822528
     0.91595904 0.92479424 0.90614144 0.91377344]
    
    随机森林 10折交叉验证平均得分: 
     0.91711776
    

    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel\__main__.py:5: DeprecationWarning: Call to deprecated `__contains__` (Method will be removed in 4.0.0, use self.wv.__contains__() instead).
    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel\__main__.py:6: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
    

    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          0
    3    7186_2          0
    4   12128_7          1
    结束.
    

### 7.Word2vec+Adaboost

* adaboost模型训练太耗时,没跑完


```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score

wv_ab_model = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1)

wv_ab_model.fit(train_data_features, label)

scores = cross_val_score(wv_ab_model, train_data_features, label, cv=3, scoring='roc_auc')

print("AdaBoost 3折交叉验证得分: \n", scores)
print("AdaBoost 3折交叉验证平均得分: \n", np.mean(scores))

test_predicted = np.array(wv_ab_model.predict(test_data_features))
print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_adaboost_wv.csv',columns = ['id','sentiment'], index = False)
print('结束.')
```

### 8.Word2vec+Voting


```python
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

wv_vot_model = VotingClassifier(estimators=[('lr', model_LR), ('xgb', xgb_model), ('rf', rf_model)],voting='hard')
wv_vot_model.fit(pd.DataFrame(train_data_features), np.array(label))

scores = cross_val_score(wv_vot_model, pd.DataFrame(train_data_features),label, cv=5, scoring=None,n_jobs = -1)

print("VotingClassifier 5折交叉验证得分: \n", scores)
print("VotingClassifier 5折交叉验证平均得分: \n", np.mean(scores))

test_predicted = np.array(vot_model.predict(pd.DataFrame(test_data_features)))
print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_vot_wv.csv',columns = ['id','sentiment'], index = False)
print('结束.')
```

    
    VotingClassifier 10折交叉验证得分: 
     [0.882  0.8712 0.892  0.8716 0.8656 0.8744 0.8776 0.876  0.8656 0.872 ]
    
    VotingClassifier 10折交叉验证平均得分: 
     0.8747999999999999
    

    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel\__main__.py:5: DeprecationWarning: Call to deprecated `__contains__` (Method will be removed in 4.0.0, use self.wv.__contains__() instead).
    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel\__main__.py:6: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\preprocessing\label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:
    

    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          1
    3    7186_2          0
    4   12128_7          1
    结束.
    

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\preprocessing\label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:
    

### 9.Word2vec+Stacking


```python
'''模型融合中使用到的各个单模型'''
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import StratifiedKFold

# 划分train数据集,调用代码,把数据集名字转成和代码一样
X = pd.DataFrame(train_data_features)
y = label

clfs = [model_LR,xgb_model,rf_model]


# 创建n_folds
n_folds = 5
skf = StratifiedKFold(y, n_folds)
print(skf)

# 创建零矩阵
dataset_blend_train = np.zeros((X.shape[0], len(clfs)))

# 建立模型
for j, clf in enumerate(clfs):
    '''依次训练各个单模型'''
    print(j, clf)
    dataset_blend_test_j = np.zeros((X_predict.shape[0], len(skf)))
    for i, (train, test) in enumerate(skf):
        '''使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。'''
        print("Fold", i)
        X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
        clf.fit(X_train, y_train)
        y_submission = clf.predict_proba(X_test)[:, 1]
        dataset_blend_train[test, j] = y_submission
        dataset_blend_test_j[:, i] = clf.predict_proba(X_predict)[:, 1]
        
    '''对于测试集，直接用这k个模型的预测值均值作为新的特征。'''
    dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

# 用建立第二层模型
stacking_model = LogisticRegression(C=0.1, max_iter=100)
stacking_model.fit(dataset_blend_train, y_train)

scores = cross_val_score(stacking_model, dataset_blend_train, label, cv=10, scoring='roc_auc')

print("stacking10折交叉验证得分:", scores)
print("stacking10折交叉验证平均得分:", np.mean(scores))
```

# 深度学习方法

## 一、深度学习建模

### 1.LSTM


```python
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import LSTM, Embedding
from keras.models import Sequential

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
from keras.callbacks import ModelCheckpoint

MAX_SEQUENCE_LENGTH = 100 # 每条新闻最大长度
EMBEDDING_DIM = 100       # 词向量空间维度

all_data = train_data+test_data
#Tokenizer是一个用于向量化文本，或将文本转换为序列（即单词在字典中的下标构成的列表，从1算起）的类
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_data)
sequences = tokenizer.texts_to_sequences(all_data)

#总共词数(word_index：key:词，value:索引)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
#print(word_index)

#将整篇文章根据向量化文本序列都退少补生成文章矩阵
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data.shape)

x_train,x_test = data[:len(train_data)],data[len(train_data):]

#将标签独热向量处理
labels = to_categorical(np.asarray(label))
print('Shape of label tensor:', labels.shape)


model = Sequential()
model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))#LSTM参数：LSTM的输出向量的维度
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
model.summary()

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])

VALIDATION_SPLIT = 0.16 # 验证集比例
TEST_SPLIT = 0.2 # 测试集比例

p1 = int(len(x_train)*(1-VALIDATION_SPLIT-TEST_SPLIT))
p2 = int(len(x_train)*(1-TEST_SPLIT))

train_x = x_train[:p1]
train_y = labels[:p1]
val_x = x_train[p1:p2]
val_y = labels[p1:p2]
test_x = x_train[p2:]
test_y = labels[p2:]

print ('train docs: '+str(len(train_x)))
print ('val docs: '+str(len(val_x)))
print ('test docs: '+str(len(test_x)))

filepath="lstm_model/lstm_weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,mode='max')
callbacks_list = [checkpoint]

# Fit the model
#model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10,callbacks=callbacks_list, verbose=0)

model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=12, batch_size=5000,callbacks=callbacks_list)

#model.save('word_vector_cnn.h5')
print (model.evaluate(test_x, test_y))

test_predicted = np.array(model.predict_classes(x_test))

print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_lstm.csv',columns = ['id','sentiment'], index = False)
print('结束.')

```

    Found 101245 unique tokens.
    Shape of data tensor: (50000, 100)
    Shape of label tensor: (25000, 2)
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_4 (Embedding)      (None, 100, 100)          10124600  
    _________________________________________________________________
    lstm_5 (LSTM)                (None, 100)               80400     
    _________________________________________________________________
    dropout_6 (Dropout)          (None, 100)               0         
    _________________________________________________________________
    dense_7 (Dense)              (None, 2)                 202       
    =================================================================
    Total params: 10,205,202
    Trainable params: 10,205,202
    Non-trainable params: 0
    _________________________________________________________________
    train docs: 15999
    val docs: 4001
    test docs: 5000
    Train on 15999 samples, validate on 4001 samples
    Epoch 1/12
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.6922 - acc: 0.5401 - val_loss: 0.6898 - val_acc: 0.6151
    
    Epoch 00001: val_acc improved from -inf to 0.61510, saving model to lstm_model/lstm_weights-improvement-01-0.62.hdf5
    Epoch 2/12
    15999/15999 [==============================] - 40s 2ms/step - loss: 0.6865 - acc: 0.6664 - val_loss: 0.6823 - val_acc: 0.6776
    
    Epoch 00002: val_acc improved from 0.61510 to 0.67758, saving model to lstm_model/lstm_weights-improvement-02-0.68.hdf5
    Epoch 3/12
    15999/15999 [==============================] - 38s 2ms/step - loss: 0.6728 - acc: 0.7438 - val_loss: 0.6590 - val_acc: 0.7423
    
    Epoch 00003: val_acc improved from 0.67758 to 0.74231, saving model to lstm_model/lstm_weights-improvement-03-0.74.hdf5
    Epoch 4/12
    15999/15999 [==============================] - 47s 3ms/step - loss: 0.6261 - acc: 0.8066 - val_loss: 0.5257 - val_acc: 0.7908
    
    Epoch 00004: val_acc improved from 0.74231 to 0.79080, saving model to lstm_model/lstm_weights-improvement-04-0.79.hdf5
    Epoch 5/12
    15999/15999 [==============================] - 46s 3ms/step - loss: 0.5113 - acc: 0.7992 - val_loss: 0.4856 - val_acc: 0.8180
    
    Epoch 00005: val_acc improved from 0.79080 to 0.81805, saving model to lstm_model/lstm_weights-improvement-05-0.82.hdf5
    Epoch 6/12
    15999/15999 [==============================] - 40s 2ms/step - loss: 0.4427 - acc: 0.8469 - val_loss: 0.3895 - val_acc: 0.8460
    
    Epoch 00006: val_acc improved from 0.81805 to 0.84604, saving model to lstm_model/lstm_weights-improvement-06-0.85.hdf5
    Epoch 7/12
    15999/15999 [==============================] - 44s 3ms/step - loss: 0.3473 - acc: 0.8820 - val_loss: 0.3471 - val_acc: 0.8583
    
    Epoch 00007: val_acc improved from 0.84604 to 0.85829, saving model to lstm_model/lstm_weights-improvement-07-0.86.hdf5
    Epoch 8/12
    15999/15999 [==============================] - 46s 3ms/step - loss: 0.2877 - acc: 0.9022 - val_loss: 0.3206 - val_acc: 0.8670
    
    Epoch 00008: val_acc improved from 0.85829 to 0.86703, saving model to lstm_model/lstm_weights-improvement-08-0.87.hdf5
    Epoch 9/12
    15999/15999 [==============================] - 46s 3ms/step - loss: 0.2308 - acc: 0.9202 - val_loss: 0.3152 - val_acc: 0.8693
    
    Epoch 00009: val_acc improved from 0.86703 to 0.86928, saving model to lstm_model/lstm_weights-improvement-09-0.87.hdf5
    Epoch 10/12
    15999/15999 [==============================] - 46s 3ms/step - loss: 0.1895 - acc: 0.9388 - val_loss: 0.3132 - val_acc: 0.8735
    
    Epoch 00010: val_acc improved from 0.86928 to 0.87353, saving model to lstm_model/lstm_weights-improvement-10-0.87.hdf5
    Epoch 11/12
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.1540 - acc: 0.9499 - val_loss: 0.3206 - val_acc: 0.8725
    
    Epoch 00011: val_acc did not improve from 0.87353
    Epoch 12/12
    15999/15999 [==============================] - 38s 2ms/step - loss: 0.1284 - acc: 0.9604 - val_loss: 0.3267 - val_acc: 0.8750
    
    Epoch 00012: val_acc improved from 0.87353 to 0.87503, saving model to lstm_model/lstm_weights-improvement-12-0.88.hdf5
    5000/5000 [==============================] - 2s 496us/step
    [0.3418441625118256, 0.866]
    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          0
    3    7186_2          1
    4   12128_7          1
    结束.
    

### 2.CNN


```python
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Sequential
from keras.utils import plot_model

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import numpy as np


MAX_SEQUENCE_LENGTH = 100 # 每条新闻最大长度
EMBEDDING_DIM = 100 # 词向量空间维度



#合并训练集和测试集
all_data = train_data+test_data

#Tokenizer是一个用于向量化文本，或将文本转换为序列（即单词在字典中的下标构成的列表，从1算起）的类
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_data)
sequences = tokenizer.texts_to_sequences(all_data)

#总共词数(word_index：key:词，value:索引)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
#print(word_index)

#将整篇文章根据向量化文本序列都退少补生成文章矩阵
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data.shape)

x_train,x_test = data[:len(train_data)],data[len(train_data):]

#将标签独热向量处理
labels = to_categorical(np.asarray(label))
print('Shape of label tensor:', labels.shape)

VALIDATION_SPLIT = 0.16 # 验证集比例
TEST_SPLIT = 0.2 # 测试集比例

p1 = int(len(x_train)*(1-VALIDATION_SPLIT-TEST_SPLIT))
p2 = int(len(x_train)*(1-TEST_SPLIT))

train_x = x_train[:p1]
train_y = labels[:p1]
val_x = x_train[p1:p2]
val_y = labels[p1:p2]
test_x = x_train[p2:]
test_y = labels[p2:]

print ('train docs: '+str(len(train_x)))
print ('val docs: '+str(len(val_x)))
print ('test docs: '+str(len(test_x)))

model = Sequential()
model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(Dropout(0.2))
model.add(Conv1D(250, 3, padding='valid', activation='relu', strides=1))
model.add(MaxPooling1D(3))
model.add(Flatten())
model.add(Dense(EMBEDDING_DIM, activation='relu'))
model.add(Dense(labels.shape[1], activation='softmax'))
model.summary()

#plot_model(model, to_file='model.png',show_shapes=True)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])


filepath="cnn_weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,mode='max')
callbacks_list = [checkpoint]

# Fit the model
#model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10,callbacks=callbacks_list, verbose=0)


model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=12, batch_size=5000,callbacks=callbacks_list)

#model.save('word_vector_cnn.h5')
print (model.evaluate(test_x, test_y))

test_predicted = np.array(model.predict_classes(x_test))

print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_cnn.csv',columns = ['id','sentiment'], index = False)
print('结束.')

```

    C:\ProgramData\Anaconda3\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.
    

    Found 101245 unique tokens.
    Shape of data tensor: (50000, 100)
    Shape of label tensor: (25000, 2)
    train docs: 15999
    val docs: 4001
    test docs: 5000
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 100, 100)          10124600  
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 100, 100)          0         
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 98, 250)           75250     
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 32, 250)           0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 8000)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 100)               800100    
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 202       
    =================================================================
    Total params: 11,000,152
    Trainable params: 11,000,152
    Non-trainable params: 0
    _________________________________________________________________
    Train on 15999 samples, validate on 4001 samples
    Epoch 1/6
    15000/15999 [===========================>..] - ETA: 21s - loss: 1.0420 - acc: 0.5028 

    C:\ProgramData\Anaconda3\lib\site-packages\keras\callbacks.py:122: UserWarning: Method on_batch_end() is slow compared to the batch update (5.411809). Check your callbacks.
      % delta_t_median)
    

    15999/15999 [==============================] - 360s 22ms/step - loss: 1.0201 - acc: 0.5049 - val_loss: 0.6928 - val_acc: 0.4989
    
    Epoch 00001: val_acc improved from -inf to 0.49888, saving model to weights-improvement-01-0.50.hdf5
    Epoch 2/6
    15999/15999 [==============================] - 125s 8ms/step - loss: 0.6877 - acc: 0.5836 - val_loss: 0.6910 - val_acc: 0.4986
    
    Epoch 00002: val_acc did not improve from 0.49888
    Epoch 3/6
    15999/15999 [==============================] - 36s 2ms/step - loss: 0.6810 - acc: 0.5130 - val_loss: 0.6879 - val_acc: 0.4986
    
    Epoch 00003: val_acc did not improve from 0.49888
    Epoch 4/6
    15999/15999 [==============================] - 33s 2ms/step - loss: 0.6656 - acc: 0.5419 - val_loss: 0.6849 - val_acc: 0.4994
    
    Epoch 00004: val_acc improved from 0.49888 to 0.49938, saving model to weights-improvement-04-0.50.hdf5
    Epoch 5/6
    15999/15999 [==============================] - 56s 4ms/step - loss: 0.6498 - acc: 0.5272 - val_loss: 0.6014 - val_acc: 0.7183
    
    Epoch 00005: val_acc improved from 0.49938 to 0.71832, saving model to weights-improvement-05-0.72.hdf5
    Epoch 6/6
    15999/15999 [==============================] - 34s 2ms/step - loss: 0.5499 - acc: 0.7414 - val_loss: 0.5173 - val_acc: 0.7471
    
    Epoch 00006: val_acc improved from 0.71832 to 0.74706, saving model to weights-improvement-06-0.75.hdf5
    5000/5000 [==============================] - 4s 865us/step
    [0.5218041631698609, 0.7492]
    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          1
    3    7186_2          1
    4   12128_7          1
    结束.
    

## 二、Word2vec+深度学习建模

### 1.LSTM+Word2Vec


```python
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import LSTM, Embedding
from keras.models import Sequential
from keras.utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import gensim
from gensim.models.word2vec import Word2Vec
from keras.callbacks import ModelCheckpoint


all_data = train_data+test_data
#Tokenizer是一个用于向量化文本，或将文本转换为序列（即单词在字典中的下标构成的列表，从1算起）的类
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_data)
sequences = tokenizer.texts_to_sequences(all_data)

#总共词数(word_index：key:词，value:索引)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
#print(word_index)

#将整篇文章根据向量化文本序列都退少补生成文章矩阵
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data.shape)

x_train,x_test = data[:len(train_data)],data[len(train_data):]

#将标签独热向量处理
labels = to_categorical(np.asarray(label))
print('Shape of label tensor:', labels.shape)

wv_model = Word2Vec.load("100size_3min_count_10window.model")

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items(): 
    if word in wv_model:
        embedding_matrix[i] = np.asarray(wv_model[word],dtype='float32')
        
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)#冻结嵌入层

model = Sequential()
model.add(embedding_layer)
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))#LSTM参数：LSTM的输出向量的维度
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
model.summary()

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])


VALIDATION_SPLIT = 0.16 # 验证集比例
TEST_SPLIT = 0.2 # 测试集比例

p1 = int(len(x_train)*(1-VALIDATION_SPLIT-TEST_SPLIT))
p2 = int(len(x_train)*(1-TEST_SPLIT))

train_x = x_train[:p1]
train_y = labels[:p1]
val_x = x_train[p1:p2]
val_y = labels[p1:p2]
test_x = x_train[p2:]
test_y = labels[p2:]

print ('train docs: '+str(len(train_x)))
print ('val docs: '+str(len(val_x)))
print ('test docs: '+str(len(test_x)))

filepath="lstm_word2vec_model/lstm_word2vec_weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
#仅保存最好的模型
#filepath="weights.best.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,mode='max')#验证集准确率比之前效果好就保存权重
callbacks_list = [checkpoint]

# Fit the model
#model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10,callbacks=callbacks_list, verbose=0)

model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=40, batch_size=5000,callbacks=callbacks_list)

#model.save('word_vector_cnn.h5')
print (model.evaluate(test_x, test_y))


test_predicted = np.array(model.predict_classes(x_test))

print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_lstm_word2vec.csv',columns = ['id','sentiment'], index = False)
print('结束.')

```

    Found 101245 unique tokens.
    Shape of data tensor: (50000, 100)
    Shape of label tensor: (25000, 2)
    

    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel\__main__.py:39: DeprecationWarning: Call to deprecated `__contains__` (Method will be removed in 4.0.0, use self.wv.__contains__() instead).
    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel\__main__.py:40: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
    

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_2 (Embedding)      (None, 100, 100)          10124600  
    _________________________________________________________________
    lstm_2 (LSTM)                (None, 100)               80400     
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 100)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 202       
    =================================================================
    Total params: 10,205,202
    Trainable params: 80,602
    Non-trainable params: 10,124,600
    _________________________________________________________________
    train docs: 15999
    val docs: 4001
    test docs: 5000
    Train on 15999 samples, validate on 4001 samples
    Epoch 1/40
    15999/15999 [==============================] - 36s 2ms/step - loss: 0.6852 - acc: 0.5797 - val_loss: 0.6684 - val_acc: 0.7096
    
    Epoch 00001: val_acc improved from -inf to 0.70957, saving model to lstm_word2vec_weights-improvement-01-0.71.hdf5
    Epoch 2/40
    15999/15999 [==============================] - 27s 2ms/step - loss: 0.6619 - acc: 0.6885 - val_loss: 0.6328 - val_acc: 0.7546
    
    Epoch 00002: val_acc improved from 0.70957 to 0.75456, saving model to lstm_word2vec_weights-improvement-02-0.75.hdf5
    Epoch 3/40
    15999/15999 [==============================] - 29s 2ms/step - loss: 0.6155 - acc: 0.7386 - val_loss: 0.5217 - val_acc: 0.7928
    
    Epoch 00003: val_acc improved from 0.75456 to 0.79280, saving model to lstm_word2vec_weights-improvement-03-0.79.hdf5
    Epoch 4/40
    15999/15999 [==============================] - 30s 2ms/step - loss: 0.4992 - acc: 0.7850 - val_loss: 0.4086 - val_acc: 0.8230
    
    Epoch 00004: val_acc improved from 0.79280 to 0.82304, saving model to lstm_word2vec_weights-improvement-04-0.82.hdf5
    Epoch 5/40
    15999/15999 [==============================] - 30s 2ms/step - loss: 0.4629 - acc: 0.8022 - val_loss: 0.4273 - val_acc: 0.8295
    
    Epoch 00005: val_acc improved from 0.82304 to 0.82954, saving model to lstm_word2vec_weights-improvement-05-0.83.hdf5
    Epoch 6/40
    15999/15999 [==============================] - 29s 2ms/step - loss: 0.4421 - acc: 0.8135 - val_loss: 0.3680 - val_acc: 0.8515
    
    Epoch 00006: val_acc improved from 0.82954 to 0.85154, saving model to lstm_word2vec_weights-improvement-06-0.85.hdf5
    Epoch 7/40
    15999/15999 [==============================] - 30s 2ms/step - loss: 0.4188 - acc: 0.8216 - val_loss: 0.3469 - val_acc: 0.8575
    
    Epoch 00007: val_acc improved from 0.85154 to 0.85754, saving model to lstm_word2vec_weights-improvement-07-0.86.hdf5
    Epoch 8/40
    15999/15999 [==============================] - 29s 2ms/step - loss: 0.4013 - acc: 0.8312 - val_loss: 0.3650 - val_acc: 0.8568
    
    Epoch 00008: val_acc did not improve from 0.85754
    Epoch 9/40
    15999/15999 [==============================] - 29s 2ms/step - loss: 0.3988 - acc: 0.8309 - val_loss: 0.3352 - val_acc: 0.8628
    
    Epoch 00009: val_acc improved from 0.85754 to 0.86278, saving model to lstm_word2vec_weights-improvement-09-0.86.hdf5
    Epoch 10/40
    15999/15999 [==============================] - 29s 2ms/step - loss: 0.3877 - acc: 0.8385 - val_loss: 0.3276 - val_acc: 0.8660
    
    Epoch 00010: val_acc improved from 0.86278 to 0.86603, saving model to lstm_word2vec_weights-improvement-10-0.87.hdf5
    Epoch 11/40
    15999/15999 [==============================] - 29s 2ms/step - loss: 0.3876 - acc: 0.8387 - val_loss: 0.3492 - val_acc: 0.8590
    
    Epoch 00011: val_acc did not improve from 0.86603
    Epoch 12/40
    15999/15999 [==============================] - 32s 2ms/step - loss: 0.3873 - acc: 0.8377 - val_loss: 0.3324 - val_acc: 0.8635
    
    Epoch 00012: val_acc did not improve from 0.86603
    Epoch 13/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3783 - acc: 0.8371 - val_loss: 0.3253 - val_acc: 0.8640
    
    Epoch 00013: val_acc did not improve from 0.86603
    Epoch 14/40
    15999/15999 [==============================] - 40s 2ms/step - loss: 0.3739 - acc: 0.8438 - val_loss: 0.3292 - val_acc: 0.8613
    
    Epoch 00014: val_acc did not improve from 0.86603
    Epoch 15/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3735 - acc: 0.8406 - val_loss: 0.3234 - val_acc: 0.8665
    
    Epoch 00015: val_acc improved from 0.86603 to 0.86653, saving model to lstm_word2vec_weights-improvement-15-0.87.hdf5
    Epoch 16/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3714 - acc: 0.8422 - val_loss: 0.3225 - val_acc: 0.8668
    
    Epoch 00016: val_acc improved from 0.86653 to 0.86678, saving model to lstm_word2vec_weights-improvement-16-0.87.hdf5
    Epoch 17/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3662 - acc: 0.8457 - val_loss: 0.3213 - val_acc: 0.8673
    
    Epoch 00017: val_acc improved from 0.86678 to 0.86728, saving model to lstm_word2vec_weights-improvement-17-0.87.hdf5
    Epoch 18/40
    15999/15999 [==============================] - 38s 2ms/step - loss: 0.3721 - acc: 0.8429 - val_loss: 0.3255 - val_acc: 0.8645
    
    Epoch 00018: val_acc did not improve from 0.86728
    Epoch 19/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3736 - acc: 0.8409 - val_loss: 0.3223 - val_acc: 0.8653
    
    Epoch 00019: val_acc did not improve from 0.86728
    Epoch 20/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3659 - acc: 0.8434 - val_loss: 0.3215 - val_acc: 0.8668
    
    Epoch 00020: val_acc did not improve from 0.86728
    Epoch 21/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3648 - acc: 0.8464 - val_loss: 0.3167 - val_acc: 0.8693
    
    Epoch 00021: val_acc improved from 0.86728 to 0.86928, saving model to lstm_word2vec_weights-improvement-21-0.87.hdf5
    Epoch 22/40
    15999/15999 [==============================] - 33s 2ms/step - loss: 0.3711 - acc: 0.8407 - val_loss: 0.3228 - val_acc: 0.8653
    
    Epoch 00022: val_acc did not improve from 0.86928
    Epoch 23/40
    15999/15999 [==============================] - 32s 2ms/step - loss: 0.3659 - acc: 0.8459 - val_loss: 0.3155 - val_acc: 0.8685
    
    Epoch 00023: val_acc did not improve from 0.86928
    Epoch 24/40
    15999/15999 [==============================] - 37s 2ms/step - loss: 0.3635 - acc: 0.8479 - val_loss: 0.3183 - val_acc: 0.8673
    
    Epoch 00024: val_acc did not improve from 0.86928
    Epoch 25/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3599 - acc: 0.8458 - val_loss: 0.3164 - val_acc: 0.8698
    
    Epoch 00025: val_acc improved from 0.86928 to 0.86978, saving model to lstm_word2vec_weights-improvement-25-0.87.hdf5
    Epoch 26/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3630 - acc: 0.8477 - val_loss: 0.3216 - val_acc: 0.8650
    
    Epoch 00026: val_acc did not improve from 0.86978
    Epoch 27/40
    15999/15999 [==============================] - 40s 2ms/step - loss: 0.3627 - acc: 0.8472 - val_loss: 0.3142 - val_acc: 0.8715
    
    Epoch 00027: val_acc improved from 0.86978 to 0.87153, saving model to lstm_word2vec_weights-improvement-27-0.87.hdf5
    Epoch 28/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3618 - acc: 0.8476 - val_loss: 0.3188 - val_acc: 0.8660
    
    Epoch 00028: val_acc did not improve from 0.87153
    Epoch 29/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3612 - acc: 0.8477 - val_loss: 0.3150 - val_acc: 0.8688
    
    Epoch 00029: val_acc did not improve from 0.87153
    Epoch 30/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3506 - acc: 0.8524 - val_loss: 0.3130 - val_acc: 0.8715
    
    Epoch 00030: val_acc did not improve from 0.87153
    Epoch 31/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3590 - acc: 0.8481 - val_loss: 0.3235 - val_acc: 0.8640
    
    Epoch 00031: val_acc did not improve from 0.87153
    Epoch 32/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3596 - acc: 0.8474 - val_loss: 0.3114 - val_acc: 0.8718
    
    Epoch 00032: val_acc improved from 0.87153 to 0.87178, saving model to lstm_word2vec_weights-improvement-32-0.87.hdf5
    Epoch 33/40
    15999/15999 [==============================] - 38s 2ms/step - loss: 0.3595 - acc: 0.8480 - val_loss: 0.3313 - val_acc: 0.8583
    
    Epoch 00033: val_acc did not improve from 0.87178
    Epoch 34/40
    15999/15999 [==============================] - 34s 2ms/step - loss: 0.3598 - acc: 0.8461 - val_loss: 0.3076 - val_acc: 0.8733
    
    Epoch 00034: val_acc improved from 0.87178 to 0.87328, saving model to lstm_word2vec_weights-improvement-34-0.87.hdf5
    Epoch 35/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3564 - acc: 0.8486 - val_loss: 0.3217 - val_acc: 0.8643
    
    Epoch 00035: val_acc did not improve from 0.87328
    Epoch 36/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3522 - acc: 0.8522 - val_loss: 0.3051 - val_acc: 0.8740
    
    Epoch 00036: val_acc improved from 0.87328 to 0.87403, saving model to lstm_word2vec_weights-improvement-36-0.87.hdf5
    Epoch 37/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3558 - acc: 0.8502 - val_loss: 0.3213 - val_acc: 0.8663
    
    Epoch 00037: val_acc did not improve from 0.87403
    Epoch 38/40
    15999/15999 [==============================] - 36s 2ms/step - loss: 0.3533 - acc: 0.8494 - val_loss: 0.3094 - val_acc: 0.8713
    
    Epoch 00038: val_acc did not improve from 0.87403
    Epoch 39/40
    15999/15999 [==============================] - 35s 2ms/step - loss: 0.3554 - acc: 0.8499 - val_loss: 0.3129 - val_acc: 0.8710
    
    Epoch 00039: val_acc did not improve from 0.87403
    Epoch 40/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3502 - acc: 0.8516 - val_loss: 0.3079 - val_acc: 0.8710
    
    Epoch 00040: val_acc did not improve from 0.87403
    5000/5000 [==============================] - 3s 636us/step
    [0.31709566490650176, 0.868]
    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          1
    3    7186_2          0
    4   12128_7          1
    结束.
    

#### 加载模型（使用保存的模型评估或继续训练）


```python
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import LSTM, Embedding
from keras.models import Sequential
from keras.utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import gensim
from gensim.models.word2vec import Word2Vec
from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(embedding_layer)
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))#LSTM参数：LSTM的输出向量的维度
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# load weights
model.load_weights("lstm_word2vec_weights-improvement-36-0.87.hdf5")

#model.save('word_vector_cnn.h5')
print (model.evaluate(test_x, test_y))
```

    5000/5000 [==============================] - 3s 519us/step
    [0.31665992724895475, 0.8666]
    

### 2.CNN+Word2Vec


```python
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Sequential
from keras.utils import plot_model

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
from gensim.models.word2vec import Word2Vec

MAX_SEQUENCE_LENGTH = 100 # 每条新闻最大长度
EMBEDDING_DIM = 100       # 词向量空间维度

VALIDATION_SPLIT = 0.16 # 验证集比例
TEST_SPLIT = 0.2 # 测试集比例

#合并训练集和测试集
all_data = train_data+test_data

#Tokenizer是一个用于向量化文本，或将文本转换为序列（即单词在字典中的下标构成的列表，从1算起）的类
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_data)
sequences = tokenizer.texts_to_sequences(all_data)

#总共词数(word_index：key:词，value:索引)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
#print(word_index)

#将整篇文章根据向量化文本序列都退少补生成文章矩阵
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data.shape)

x_train,x_test = data[:len(train_data)],data[len(train_data):]

#将标签独热向量处理
labels = to_categorical(np.asarray(label))
print('Shape of label tensor:', labels.shape)


p1 = int(len(x_train)*(1-VALIDATION_SPLIT-TEST_SPLIT))
p2 = int(len(x_train)*(1-TEST_SPLIT))

train_x = x_train[:p1]
train_y = labels[:p1]
val_x = x_train[p1:p2]
val_y = labels[p1:p2]
test_x = x_train[p2:]
test_y = labels[p2:]

print ('train docs: '+str(len(train_x)))
print ('val docs: '+str(len(val_x)))
print ('test docs: '+str(len(test_x)))


wv_model = Word2Vec.load("100size_3min_count_10window.model")

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items(): 
    if word in wv_model:
        embedding_matrix[i] = np.asarray(wv_model[word],dtype='float32')
        
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)#冻结嵌入层


model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.2))
model.add(Conv1D(250, 3, padding='valid', activation='relu', strides=1))
model.add(MaxPooling1D(3))
model.add(Flatten())
model.add(Dense(EMBEDDING_DIM, activation='relu'))
model.add(Dense(labels.shape[1], activation='softmax'))
model.summary()

#plot_model(model, to_file='model.png',show_shapes=True)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])


filepath="cnn_model/cnn_word2vec_weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,mode='max')
callbacks_list = [checkpoint]

# Fit the model
#model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10,callbacks=callbacks_list, verbose=0)


model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=20, batch_size=5000,callbacks=callbacks_list)

#model.save('word_vector_cnn.h5')
print (model.evaluate(test_x, test_y))

test_predicted = np.array(model.predict_classes(x_test))

print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_cnn_word2vec.csv',columns = ['id','sentiment'], index = False)
print('结束.')

```

    Found 101245 unique tokens.
    Shape of data tensor: (50000, 100)
    Shape of label tensor: (25000, 2)
    train docs: 15999
    val docs: 4001
    test docs: 5000
    

    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel\__main__.py:61: DeprecationWarning: Call to deprecated `__contains__` (Method will be removed in 4.0.0, use self.wv.__contains__() instead).
    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel\__main__.py:62: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
    

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_3 (Embedding)      (None, 100, 100)          10124600  
    _________________________________________________________________
    dropout_5 (Dropout)          (None, 100, 100)          0         
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 98, 250)           75250     
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 32, 250)           0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 8000)              0         
    _________________________________________________________________
    dense_5 (Dense)              (None, 100)               800100    
    _________________________________________________________________
    dense_6 (Dense)              (None, 2)                 202       
    =================================================================
    Total params: 11,000,152
    Trainable params: 875,552
    Non-trainable params: 10,124,600
    _________________________________________________________________
    Train on 15999 samples, validate on 4001 samples
    Epoch 1/20
    15999/15999 [==============================] - 18s 1ms/step - loss: 1.8571 - acc: 0.5046 - val_loss: 0.6610 - val_acc: 0.6353
    
    Epoch 00001: val_acc improved from -inf to 0.63534, saving model to cnn_model/cnn_word2vec_weights-improvement-01-0.64.hdf5
    Epoch 2/20
    15999/15999 [==============================] - 19s 1ms/step - loss: 0.6381 - acc: 0.6811 - val_loss: 0.6190 - val_acc: 0.6473
    
    Epoch 00002: val_acc improved from 0.63534 to 0.64734, saving model to cnn_model/cnn_word2vec_weights-improvement-02-0.65.hdf5
    Epoch 3/20
    15999/15999 [==============================] - 19s 1ms/step - loss: 0.6863 - acc: 0.5880 - val_loss: 0.5875 - val_acc: 0.6901
    
    Epoch 00003: val_acc improved from 0.64734 to 0.69008, saving model to cnn_model/cnn_word2vec_weights-improvement-03-0.69.hdf5
    Epoch 4/20
    15999/15999 [==============================] - 20s 1ms/step - loss: 0.5728 - acc: 0.6988 - val_loss: 0.5574 - val_acc: 0.7083
    
    Epoch 00004: val_acc improved from 0.69008 to 0.70832, saving model to cnn_model/cnn_word2vec_weights-improvement-04-0.71.hdf5
    Epoch 5/20
    15999/15999 [==============================] - 19s 1ms/step - loss: 0.5576 - acc: 0.7026 - val_loss: 0.5020 - val_acc: 0.7538
    
    Epoch 00005: val_acc improved from 0.70832 to 0.75381, saving model to cnn_model/cnn_word2vec_weights-improvement-05-0.75.hdf5
    Epoch 6/20
    15999/15999 [==============================] - 19s 1ms/step - loss: 0.6535 - acc: 0.6495 - val_loss: 0.4529 - val_acc: 0.8268
    
    Epoch 00006: val_acc improved from 0.75381 to 0.82679, saving model to cnn_model/cnn_word2vec_weights-improvement-06-0.83.hdf5
    Epoch 7/20
    15999/15999 [==============================] - 19s 1ms/step - loss: 0.4425 - acc: 0.8086 - val_loss: 0.5880 - val_acc: 0.6948
    
    Epoch 00007: val_acc did not improve from 0.82679
    Epoch 8/20
    15999/15999 [==============================] - 19s 1ms/step - loss: 0.5308 - acc: 0.7394 - val_loss: 0.3934 - val_acc: 0.8383
    
    Epoch 00008: val_acc improved from 0.82679 to 0.83829, saving model to cnn_model/cnn_word2vec_weights-improvement-08-0.84.hdf5
    Epoch 9/20
    15999/15999 [==============================] - 19s 1ms/step - loss: 0.4318 - acc: 0.8035 - val_loss: 0.4658 - val_acc: 0.7731
    
    Epoch 00009: val_acc did not improve from 0.83829
    Epoch 10/20
    15999/15999 [==============================] - 17s 1ms/step - loss: 0.4140 - acc: 0.8127 - val_loss: 0.3626 - val_acc: 0.8423
    
    Epoch 00010: val_acc improved from 0.83829 to 0.84229, saving model to cnn_model/cnn_word2vec_weights-improvement-10-0.84.hdf5
    Epoch 11/20
    15999/15999 [==============================] - 16s 990us/step - loss: 0.3769 - acc: 0.8356 - val_loss: 0.4600 - val_acc: 0.7823
    
    Epoch 00011: val_acc did not improve from 0.84229
    Epoch 12/20
    15999/15999 [==============================] - 15s 943us/step - loss: 0.4385 - acc: 0.7902 - val_loss: 0.3469 - val_acc: 0.8540
    
    Epoch 00012: val_acc improved from 0.84229 to 0.85404, saving model to cnn_model/cnn_word2vec_weights-improvement-12-0.85.hdf5
    Epoch 13/20
    15999/15999 [==============================] - 15s 944us/step - loss: 0.3454 - acc: 0.8516 - val_loss: 0.3733 - val_acc: 0.8293
    
    Epoch 00013: val_acc did not improve from 0.85404
    Epoch 14/20
    15999/15999 [==============================] - 15s 944us/step - loss: 0.4367 - acc: 0.7958 - val_loss: 0.3613 - val_acc: 0.8423
    
    Epoch 00014: val_acc did not improve from 0.85404
    Epoch 15/20
    15999/15999 [==============================] - 15s 941us/step - loss: 0.3444 - acc: 0.8539 - val_loss: 0.3234 - val_acc: 0.8598
    
    Epoch 00015: val_acc improved from 0.85404 to 0.85979, saving model to cnn_model/cnn_word2vec_weights-improvement-15-0.86.hdf5
    Epoch 16/20
    15999/15999 [==============================] - 15s 944us/step - loss: 0.3219 - acc: 0.8628 - val_loss: 0.3491 - val_acc: 0.8450
    
    Epoch 00016: val_acc did not improve from 0.85979
    Epoch 17/20
    15999/15999 [==============================] - 15s 946us/step - loss: 0.4664 - acc: 0.7804 - val_loss: 0.3365 - val_acc: 0.8595
    
    Epoch 00017: val_acc did not improve from 0.85979
    Epoch 18/20
    15999/15999 [==============================] - 15s 953us/step - loss: 0.3263 - acc: 0.8640 - val_loss: 0.3366 - val_acc: 0.8535
    
    Epoch 00018: val_acc did not improve from 0.85979
    Epoch 19/20
    15999/15999 [==============================] - 15s 940us/step - loss: 0.3643 - acc: 0.8394 - val_loss: 0.3416 - val_acc: 0.8503
    
    Epoch 00019: val_acc did not improve from 0.85979
    Epoch 20/20
    15999/15999 [==============================] - 15s 956us/step - loss: 0.3400 - acc: 0.8516 - val_loss: 0.3439 - val_acc: 0.8485
    
    Epoch 00020: val_acc did not improve from 0.85979
    5000/5000 [==============================] - 2s 404us/step
    [0.35134012341499327, 0.8474]
    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          1
    3    7186_2          0
    4   12128_7          1
    结束.
    
