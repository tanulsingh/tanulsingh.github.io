# About this Notebook

NLP is a very hot topic right now and as belived by many experts '2020 is going to be NLP's Year' ,with its ever changing dynamics it is experiencing a boom , same as computer vision once did. Owing to its popularity Kaggle launched two NLP competitions recently and me being a lover of this Hot topic prepared myself to join in my first Kaggle Competition.<br><br>
As I joined the competitions and since I was a complete beginner with Deep Learning Techniques for NLP, all my enthusiasm took a beating when I saw everyone Using all  kinds of BERT , everything just went over my head,I thought to quit but there is a special thing about Kaggle ,it just hooks you. I thought I have to learn someday , why not now , so I braced myself and sat on the learning curve. I wrote a kernel on the Tweet Sentiment Extraction competition that has now got a gold medal , it can be viewed here : https://www.kaggle.com/tanulsingh077/twitter-sentiment-extaction-analysis-eda-and-model <br><br>
After 10 days of extensive learning(finishing all the latest NLP approaches) , I am back here to share my leaning , by writing a kernel that starts from the very Basic RNN's to built over , all the way to BERT . I invite you all to come and learn alongside with me and take a step closer towards becoming an NLP expert

# Contents

In this Notebook I will start with the very Basics of RNN's and Build all the way to latest deep learning architectures to solve NLP problems. It will cover the Following:
* Simple RNN's
* Word Embeddings : Definition and How to get them
* LSTM's
* GRU's
* BI-Directional RNN's
* Encoder-Decoder Models (Seq2Seq Models)
* Attention Models
* Transformers - Attention is all you need
* BERT

I will divide every Topic into four subsections:
* Basic Overview
* In-Depth Understanding : In this I will attach links of articles and videos to learn about the topic in depth
* Code-Implementation
* Code Explanation

This is a comprehensive kernel and if you follow along till the end , I promise you would learn all the techniques completely

Note that the aim of this notebook is not to have a High LB score but to present a beginner guide to understand Deep Learning techniques used for NLP. Also after discussing all of these ideas , I will present a starter solution for this competiton

**<span style="color:Red">This kernel has been a work of more than 10 days If you find my kernel useful and my efforts appreciable, Please Upvote it , it motivates me to write more Quality content**


```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU,SimpleRNN
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping


import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
```

    Using TensorFlow backend.


# Configuring TPU's

For this version of Notebook we will be using TPU's as we have to built a BERT Model


```python
# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)
```

    Running on TPU  grpc://10.0.0.2:8470
    REPLICAS:  8



```python
train = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')
validation = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
```

We will drop the other columns and approach this problem as a Binary Classification Problem and also we will have our exercise done on a smaller subsection of the dataset(only 12000 data points) to make it easier to train the models


```python
train.drop(['severe_toxic','obscene','threat','insult','identity_hate'],axis=1,inplace=True)
```


```python
train = train.loc[:12000,:]
train.shape
```




    (12001, 3)



We will check the maximum number of words that can be present in a comment , this will help us in padding later


```python
train['comment_text'].apply(lambda x:len(str(x).split())).max()
```




    1403



Writing a function for getting auc score for validation


```python
def roc_auc(predictions,target):
    '''
    This methods returns the AUC Score when given the Predictions
    and Labels
    '''
    
    fpr, tpr, thresholds = metrics.roc_curve(target, predictions)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc
```

### Data Preparation


```python
xtrain, xvalid, ytrain, yvalid = train_test_split(train.comment_text.values, train.toxic.values, 
                                                  stratify=train.toxic.values, 
                                                  random_state=42, 
                                                  test_size=0.2, shuffle=True)
```

# Before We Begin

Before we Begin If you are a complete starter with NLP and never worked with text data, I am attaching a few kernels that will serve as a starting point of your journey
* https://www.kaggle.com/arthurtok/spooky-nlp-and-topic-modelling-tutorial
* https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle

If you want a more basic dataset to practice with here is another kernel which I wrote:
* https://www.kaggle.com/tanulsingh077/what-s-cooking

Below are some Resources to get started with basic level Neural Networks, It will help us to easily understand the upcoming parts
* https://www.youtube.com/watch?v=aircAruvnKk&list=PL_h2yd2CGtBHEKwEH5iqTZH85wLS-eUzv
* https://www.youtube.com/watch?v=IHZwWFHWa-w&list=PL_h2yd2CGtBHEKwEH5iqTZH85wLS-eUzv&index=2
* https://www.youtube.com/watch?v=Ilg3gGewQ5U&list=PL_h2yd2CGtBHEKwEH5iqTZH85wLS-eUzv&index=3
* https://www.youtube.com/watch?v=tIeHLnjs5U8&list=PL_h2yd2CGtBHEKwEH5iqTZH85wLS-eUzv&index=4

For Learning how to visualize test data and what to use view:
* https://www.kaggle.com/tanulsingh077/twitter-sentiment-extaction-analysis-eda-and-model
* https://www.kaggle.com/jagangupta/stop-the-s-toxic-comments-eda

# Simple RNN

## Basic Overview

What is a RNN?

Recurrent Neural Network(RNN) are a type of Neural Network where the output from previous step are fed as input to the current step. In traditional neural networks, all the inputs and outputs are independent of each other, but in cases like when it is required to predict the next word of a sentence, the previous words are required and hence there is a need to remember the previous words. Thus RNN came into existence, which solved this issue with the help of a Hidden Layer.

Why RNN's?

https://www.quora.com/Why-do-we-use-an-RNN-instead-of-a-simple-neural-network

## In-Depth Understanding

* https://medium.com/mindorks/understanding-the-recurrent-neural-network-44d593f112a2
* https://www.youtube.com/watch?v=2E65LDnM2cA&list=PL1F3ABbhcqa3BBWo170U4Ev2wfsF7FN8l
* https://www.d2l.ai/chapter_recurrent-neural-networks/rnn.html

## Code Implementation

So first I will implement the and then I will explain the code step by step


```python
# using keras tokenizer here
token = text.Tokenizer(num_words=None)
max_len = 1500

token.fit_on_texts(list(xtrain) + list(xvalid))
xtrain_seq = token.texts_to_sequences(xtrain)
xvalid_seq = token.texts_to_sequences(xvalid)

#zero pad the sequences
xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)

word_index = token.word_index
```


```python
%%time
with strategy.scope():
    # A simpleRNN without any pretrained embeddings and one dense layer
    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                     300,
                     input_length=max_len))
    model.add(SimpleRNN(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 1500, 300)         13049100  
    _________________________________________________________________
    simple_rnn_1 (SimpleRNN)     (None, 100)               40100     
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 101       
    =================================================================
    Total params: 13,089,301
    Trainable params: 13,089,301
    Non-trainable params: 0
    _________________________________________________________________
    CPU times: user 620 ms, sys: 370 ms, total: 990 ms
    Wall time: 1.18 s



```python
model.fit(xtrain_pad, ytrain, nb_epoch=5, batch_size=64*strategy.num_replicas_in_sync) #Multiplying by Strategy to run on TPU's
```

    /opt/conda/lib/python3.6/site-packages/ipykernel_launcher.py:1: UserWarning:
    
    The `nb_epoch` argument in `fit` has been renamed `epochs`.
    


    Epoch 1/5
    9600/9600 [==============================] - 39s 4ms/step - loss: 0.3714 - accuracy: 0.8805
    Epoch 2/5
    9600/9600 [==============================] - 39s 4ms/step - loss: 0.2858 - accuracy: 0.9055
    Epoch 3/5
    9600/9600 [==============================] - 40s 4ms/step - loss: 0.2748 - accuracy: 0.8945
    Epoch 4/5
    9600/9600 [==============================] - 40s 4ms/step - loss: 0.2416 - accuracy: 0.9053
    Epoch 5/5
    9600/9600 [==============================] - 39s 4ms/step - loss: 0.2109 - accuracy: 0.9079





    <keras.callbacks.callbacks.History at 0x7fae866d75c0>




```python
scores = model.predict(xvalid_pad)
print("Auc: %.2f%%" % (roc_auc(scores,yvalid)))
```

    Auc: 0.69%



```python
scores_model = []
scores_model.append({'Model': 'SimpleRNN','AUC_Score': roc_auc(scores,yvalid)})
```

## Code Explanantion
* Tokenization<br><br>
 So if you have watched the videos and referred to the links, you would know that in an RNN we input a sentence word by word. We represent every word as one hot vectors of dimensions : Numbers of words in Vocab +1. <br>
  What keras Tokenizer does is , it takes all the unique words in the corpus,forms a dictionary with words as keys and their number of occurences as values,it then sorts the dictionary in descending order of counts. It then assigns the first value 1 , second value 2 and so on. So let's suppose word 'the' occured the most in the corpus then it will assigned index 1 and vector representing 'the' would be a one-hot vector with value 1 at position 1 and rest zereos.<br>
  Try printing first 2 elements of xtrain_seq you will see every word is represented as a digit now


```python
xtrain_seq[:1]
```




    [[664,
      65,
      7,
      19,
      2262,
      14102,
      5,
      2262,
      20439,
      6071,
      4,
      71,
      32,
      20440,
      6620,
      39,
      6,
      664,
      65,
      11,
      8,
      20441,
      1502,
      38,
      6072]]



<b>Now you might be wondering What is padding? Why its done</b><br><br>

Here is the answer :
* https://www.quora.com/Which-effect-does-sequence-padding-have-on-the-training-of-a-neural-network
* https://machinelearningmastery.com/data-preparation-variable-length-input-sequences-sequence-prediction/
* https://www.coursera.org/lecture/natural-language-processing-tensorflow/padding-2Cyzs

Also sometimes people might use special tokens while tokenizing like EOS(end of string) and BOS(Begining of string). Here is the reason why it's done
* https://stackoverflow.com/questions/44579161/why-do-we-do-padding-in-nlp-tasks


The code token.word_index simply gives the dictionary of vocab that keras created for us

* Building the Neural Network

To understand the Dimensions of input and output given to RNN in keras her is a beautiful article : https://medium.com/@shivajbd/understanding-input-and-output-shape-in-lstm-keras-c501ee95c65e

The first line model.Sequential() tells keras that we will be building our network sequentially . Then we first add the Embedding layer.
Embedding layer is also a layer of neurons which takes in as input the nth dimensional one hot vector of every word and converts it into 300 dimensional vector , it gives us word embeddings similar to word2vec. We could have used word2vec but the embeddings layer learns during training to enhance the embeddings.
Next we add an 100 LSTM units without any dropout or regularization
At last we add a single neuron with sigmoid function which takes output from 100 LSTM cells (Please note we have 100 LSTM cells not layers) to predict the results and then we compile the model using adam optimizer 

* Comments on the model<br><br>
We can see our model achieves an accuracy of 1 which is just insane , we are clearly overfitting I know , but this was the simplest model of all ,we can tune a lot of hyperparameters like RNN units, we can do batch normalization , dropouts etc to get better result. The point is we got an AUC score of 0.82 without much efforts and we know have learnt about RNN's .Deep learning is really revolutionary

# Word Embeddings

While building our simple RNN models we talked about using word-embeddings , So what is word-embeddings and how do we get word-embeddings?
Here is the answer :
* https://www.coursera.org/learn/nlp-sequence-models/lecture/6Oq70/word-representation
* https://machinelearningmastery.com/what-are-word-embeddings/
<br> <br>
The latest approach to getting word Embeddings is using pretained GLoVe or using Fasttext. Without going into too much details, I would explain how to create sentence vectors and how can we use them to create a machine learning model on top of it and since I am a fan of GloVe vectors, word2vec and fasttext. In this Notebook, I'll be using the GloVe vectors. You can download the GloVe vectors from here http://www-nlp.stanford.edu/data/glove.840B.300d.zip or you can search for GloVe in datasets on Kaggle and add the file


```python
# load the GloVe vectors in a dictionary:

embeddings_index = {}
f = open('/kaggle/input/glove840b300dtxt/glove.840B.300d.txt','r',encoding='utf-8')
for line in tqdm(f):
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray([float(val) for val in values[1:]])
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
```

    2196018it [06:43, 5439.09it/s]

    Found 2196017 word vectors.


    


# LSTM's

## Basic Overview

Simple RNN's were certainly better than classical ML algorithms and gave state of the art results, but it failed to capture long term dependencies that is present in sentences . So in 1998-99 LSTM's were introduced to counter to these drawbacks.

## In Depth Understanding

Why LSTM's?
* https://www.coursera.org/learn/nlp-sequence-models/lecture/PKMRR/vanishing-gradients-with-rnns
* https://www.analyticsvidhya.com/blog/2017/12/fundamentals-of-deep-learning-introduction-to-lstm/

What are LSTM's?
* https://www.coursera.org/learn/nlp-sequence-models/lecture/KXoay/long-short-term-memory-lstm
* https://distill.pub/2019/memorization-in-rnns/
* https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21

# Code Implementation

We have already tokenized and paded our text for input to LSTM's


```python
# create an embedding matrix for the words we have in the dataset
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
```

    100%|██████████| 43496/43496 [00:00<00:00, 183357.18it/s]



```python
%%time
with strategy.scope():
    
    # A simple LSTM with glove embeddings and one dense layer
    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))

    model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    
model.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_2 (Embedding)      (None, 1500, 300)         13049100  
    _________________________________________________________________
    lstm_1 (LSTM)                (None, 100)               160400    
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 101       
    =================================================================
    Total params: 13,209,601
    Trainable params: 160,501
    Non-trainable params: 13,049,100
    _________________________________________________________________
    CPU times: user 1.33 s, sys: 1.46 s, total: 2.79 s
    Wall time: 3.09 s



```python
model.fit(xtrain_pad, ytrain, nb_epoch=5, batch_size=64*strategy.num_replicas_in_sync)
```

    /opt/conda/lib/python3.6/site-packages/ipykernel_launcher.py:1: UserWarning:
    
    The `nb_epoch` argument in `fit` has been renamed `epochs`.
    


    Epoch 1/5
    9600/9600 [==============================] - 117s 12ms/step - loss: 0.3525 - accuracy: 0.8852
    Epoch 2/5
    9600/9600 [==============================] - 114s 12ms/step - loss: 0.2397 - accuracy: 0.9192
    Epoch 3/5
    9600/9600 [==============================] - 114s 12ms/step - loss: 0.1904 - accuracy: 0.9333
    Epoch 4/5
    9600/9600 [==============================] - 114s 12ms/step - loss: 0.1659 - accuracy: 0.9394
    Epoch 5/5
    9600/9600 [==============================] - 114s 12ms/step - loss: 0.1553 - accuracy: 0.9470





    <keras.callbacks.callbacks.History at 0x7fae84dac710>




```python
scores = model.predict(xvalid_pad)
print("Auc: %.2f%%" % (roc_auc(scores,yvalid)))
```

    Auc: 0.96%



```python
scores_model.append({'Model': 'LSTM','AUC_Score': roc_auc(scores,yvalid)})
```

## Code Explanation

As a first step we calculate embedding matrix for our vocabulary from the pretrained GLoVe vectors . Then while building the embedding layer we pass Embedding Matrix as weights to the layer instead of training it over Vocabulary and thus we pass trainable = False.
Rest of the model is same as before except we have replaced the SimpleRNN By LSTM Units

* Comments on the Model

We now see that the model is not overfitting and achieves an auc score of 0.96 which is quite commendable , also we close in on the gap between accuracy and auc .
We see that in this case we used dropout and prevented overfitting the data

# GRU's

## Basic  Overview

Introduced by Cho, et al. in 2014, GRU (Gated Recurrent Unit) aims to solve the vanishing gradient problem which comes with a standard recurrent neural network. GRU's are a variation on the LSTM because both are designed similarly and, in some cases, produce equally excellent results . GRU's were designed to be simpler and faster than LSTM's and in most cases produce equally good results and thus there is no clear winner.

## In Depth Explanation

* https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be
* https://www.coursera.org/learn/nlp-sequence-models/lecture/agZiL/gated-recurrent-unit-gru
* https://www.geeksforgeeks.org/gated-recurrent-unit-networks/

## Code Implementation


```python
%%time
with strategy.scope():
    # GRU with glove embeddings and two dense layers
     model = Sequential()
     model.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
     model.add(SpatialDropout1D(0.3))
     model.add(GRU(300))
     model.add(Dense(1, activation='sigmoid'))

     model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])   
    
model.summary()
```

    Model: "sequential_3"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_3 (Embedding)      (None, 1500, 300)         13049100  
    _________________________________________________________________
    spatial_dropout1d_1 (Spatial (None, 1500, 300)         0         
    _________________________________________________________________
    gru_1 (GRU)                  (None, 300)               540900    
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 301       
    =================================================================
    Total params: 13,590,301
    Trainable params: 541,201
    Non-trainable params: 13,049,100
    _________________________________________________________________
    CPU times: user 1.3 s, sys: 1.29 s, total: 2.59 s
    Wall time: 2.79 s



```python
model.fit(xtrain_pad, ytrain, nb_epoch=5, batch_size=64*strategy.num_replicas_in_sync)
```

    /opt/conda/lib/python3.6/site-packages/ipykernel_launcher.py:1: UserWarning:
    
    The `nb_epoch` argument in `fit` has been renamed `epochs`.
    


    Epoch 1/5
    9600/9600 [==============================] - 191s 20ms/step - loss: 0.3272 - accuracy: 0.8933
    Epoch 2/5
    9600/9600 [==============================] - 189s 20ms/step - loss: 0.2015 - accuracy: 0.9334
    Epoch 3/5
    9600/9600 [==============================] - 189s 20ms/step - loss: 0.1540 - accuracy: 0.9483
    Epoch 4/5
    9600/9600 [==============================] - 189s 20ms/step - loss: 0.1287 - accuracy: 0.9548
    Epoch 5/5
    9600/9600 [==============================] - 188s 20ms/step - loss: 0.1238 - accuracy: 0.9551





    <keras.callbacks.callbacks.History at 0x7fae5b01ed30>




```python
scores = model.predict(xvalid_pad)
print("Auc: %.2f%%" % (roc_auc(scores,yvalid)))
```

    Auc: 0.97%



```python
scores_model.append({'Model': 'GRU','AUC_Score': roc_auc(scores,yvalid)})
```


```python
scores_model
```




    [{'Model': 'SimpleRNN', 'AUC_Score': 0.6949714081921305},
     {'Model': 'LSTM', 'AUC_Score': 0.9598235453841757},
     {'Model': 'GRU', 'AUC_Score': 0.9716554069114769}]



# Bi-Directional RNN's

## In Depth Explanation

* https://www.coursera.org/learn/nlp-sequence-models/lecture/fyXnn/bidirectional-rnn
* https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66
* https://d2l.ai/chapter_recurrent-modern/bi-rnn.html

## Code Implementation


```python
%%time
with strategy.scope():
    # A simple bidirectional LSTM with glove embeddings and one dense layer
    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
    model.add(Bidirectional(LSTM(300, dropout=0.3, recurrent_dropout=0.3)))

    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    
    
model.summary()
```

    Model: "sequential_4"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_4 (Embedding)      (None, 1500, 300)         13049100  
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, 600)               1442400   
    _________________________________________________________________
    dense_4 (Dense)              (None, 1)                 601       
    =================================================================
    Total params: 14,492,101
    Trainable params: 1,443,001
    Non-trainable params: 13,049,100
    _________________________________________________________________
    CPU times: user 2.39 s, sys: 1.62 s, total: 4 s
    Wall time: 3.41 s



```python
model.fit(xtrain_pad, ytrain, nb_epoch=5, batch_size=64*strategy.num_replicas_in_sync)
```

    /opt/conda/lib/python3.6/site-packages/ipykernel_launcher.py:1: UserWarning:
    
    The `nb_epoch` argument in `fit` has been renamed `epochs`.
    


    Epoch 1/5
    9600/9600 [==============================] - 322s 34ms/step - loss: 0.3171 - accuracy: 0.9009
    Epoch 2/5
    9600/9600 [==============================] - 318s 33ms/step - loss: 0.1988 - accuracy: 0.9305
    Epoch 3/5
    9600/9600 [==============================] - 318s 33ms/step - loss: 0.1650 - accuracy: 0.9424
    Epoch 4/5
    9600/9600 [==============================] - 318s 33ms/step - loss: 0.1577 - accuracy: 0.9414
    Epoch 5/5
    9600/9600 [==============================] - 319s 33ms/step - loss: 0.1540 - accuracy: 0.9459





    <keras.callbacks.callbacks.History at 0x7fae5a4ade48>




```python
scores = model.predict(xvalid_pad)
print("Auc: %.2f%%" % (roc_auc(scores,yvalid)))
```

    Auc: 0.97%



```python
scores_model.append({'Model': 'Bi-directional LSTM','AUC_Score': roc_auc(scores,yvalid)})
```

## Code Explanation

Code is same as before,only we have added bidirectional nature to the LSTM cells we used before and is self explanatory. We have achieve similar accuracy and auc score as before and now we have learned all the types of typical RNN architectures

**We are now at the end of part 1 of this notebook and things are about to go wild now as we Enter more complex and State of the art models .If you have followed along from the starting and read all the articles and understood everything , these complex models would be fairly easy to understand.I recommend Finishing Part 1 before continuing as the upcoming techniques can be quite overwhelming**

# Seq2Seq Model Architecture

## Overview

RNN's are of many types  and different architectures are used for different purposes. Here is a nice video explanining different types of model architectures : https://www.coursera.org/learn/nlp-sequence-models/lecture/BO8PS/different-types-of-rnns.
Seq2Seq is a many to many RNN architecture where the input is a sequence and the output is also a sequence (where input and output sequences can be or cannot be of different lengths). This architecture is used in a lot of applications like Machine Translation, text summarization, question answering etc

## In Depth Understanding

I will not write the code implementation for this,but rather I will provide the resources where code has already been implemented and explained in a much better way than I could have ever explained.

* https://www.coursera.org/learn/nlp-sequence-models/lecture/HyEui/basic-models ---> A basic idea of different Seq2Seq Models

* https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html , https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/ ---> Basic Encoder-Decoder Model and its explanation respectively

* https://towardsdatascience.com/how-to-implement-seq2seq-lstm-model-in-keras-shortcutnlp-6f355f3e5639 ---> A More advanced Seq2seq Model and its explanation

* https://d2l.ai/chapter_recurrent-modern/machine-translation-and-dataset.html , https://d2l.ai/chapter_recurrent-modern/encoder-decoder.html ---> Implementation of Encoder-Decoder Model from scratch

* https://www.youtube.com/watch?v=IfsjMg4fLWQ&list=PLtmWHNX-gukKocXQOkQjuVxglSDYWsSh9&index=8&t=0s ---> Introduction to Seq2seq By fast.ai


```python
# Visualization of Results obtained from various Deep learning models
results = pd.DataFrame(scores_model).sort_values(by='AUC_Score',ascending=False)
results.style.background_gradient(cmap='Blues')
```




<style  type="text/css" >
    #T_81e3fa40_7db4_11ea_96d7_0242ac130202row0_col1 {
            background-color:  #08306b;
            color:  #f1f1f1;
        }    #T_81e3fa40_7db4_11ea_96d7_0242ac130202row1_col1 {
            background-color:  #083471;
            color:  #f1f1f1;
        }    #T_81e3fa40_7db4_11ea_96d7_0242ac130202row2_col1 {
            background-color:  #083a7a;
            color:  #f1f1f1;
        }    #T_81e3fa40_7db4_11ea_96d7_0242ac130202row3_col1 {
            background-color:  #f7fbff;
            color:  #000000;
        }</style><table id="T_81e3fa40_7db4_11ea_96d7_0242ac130202" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Model</th>        <th class="col_heading level0 col1" >AUC_Score</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_81e3fa40_7db4_11ea_96d7_0242ac130202level0_row0" class="row_heading level0 row0" >2</th>
                        <td id="T_81e3fa40_7db4_11ea_96d7_0242ac130202row0_col0" class="data row0 col0" >GRU</td>
                        <td id="T_81e3fa40_7db4_11ea_96d7_0242ac130202row0_col1" class="data row0 col1" >0.971655</td>
            </tr>
            <tr>
                        <th id="T_81e3fa40_7db4_11ea_96d7_0242ac130202level0_row1" class="row_heading level0 row1" >3</th>
                        <td id="T_81e3fa40_7db4_11ea_96d7_0242ac130202row1_col0" class="data row1 col0" >Bi-directional LSTM</td>
                        <td id="T_81e3fa40_7db4_11ea_96d7_0242ac130202row1_col1" class="data row1 col1" >0.966693</td>
            </tr>
            <tr>
                        <th id="T_81e3fa40_7db4_11ea_96d7_0242ac130202level0_row2" class="row_heading level0 row2" >1</th>
                        <td id="T_81e3fa40_7db4_11ea_96d7_0242ac130202row2_col0" class="data row2 col0" >LSTM</td>
                        <td id="T_81e3fa40_7db4_11ea_96d7_0242ac130202row2_col1" class="data row2 col1" >0.959824</td>
            </tr>
            <tr>
                        <th id="T_81e3fa40_7db4_11ea_96d7_0242ac130202level0_row3" class="row_heading level0 row3" >0</th>
                        <td id="T_81e3fa40_7db4_11ea_96d7_0242ac130202row3_col0" class="data row3 col0" >SimpleRNN</td>
                        <td id="T_81e3fa40_7db4_11ea_96d7_0242ac130202row3_col1" class="data row3 col1" >0.694971</td>
            </tr>
    </tbody></table>




```python
fig = go.Figure(go.Funnelarea(
    text =results.Model,
    values = results.AUC_Score,
    title = {"position": "top center", "text": "Funnel-Chart of Sentiment Distribution"}
    ))
fig.show()
```


<script type="text/javascript">
window.PlotlyConfig = {MathJaxConfig: 'local'};
if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
if (typeof require !== 'undefined') {
require.undef("plotly");
requirejs.config({
    paths: {
        'plotly': ['https://cdn.plot.ly/plotly-latest.min']
    }
});
require(['plotly'], function(Plotly) {
    window._Plotly = Plotly;
});
}
</script>




<div>


            <div id="6bedf72d-928a-47b5-83a4-e0269d18dbf4" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("6bedf72d-928a-47b5-83a4-e0269d18dbf4")) {
                    Plotly.newPlot(
                        '6bedf72d-928a-47b5-83a4-e0269d18dbf4',
                        [{"text": ["GRU", "Bi-directional LSTM", "LSTM", "SimpleRNN"], "title": {"position": "top center", "text": "Funnel-Chart of Sentiment Distribution"}, "type": "funnelarea", "values": [0.9716554069114769, 0.9666928741352547, 0.9598235453841757, 0.6949714081921305]}],
                        {"template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('6bedf72d-928a-47b5-83a4-e0269d18dbf4');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


# Attention Models

This is the toughest and most tricky part. If you are able to understand the intiuition and working of attention block , understanding transformers and transformer based architectures like BERT will be a piece of cake. This is the part where I spent the most time on and I suggest you do the same . Please read and view the following resources in the order I am providing to ignore getting confused, also at the end of this try to write and draw an attention block in your own way :-

* https://www.coursera.org/learn/nlp-sequence-models/lecture/RDXpX/attention-model-intuition --> Only watch this video and not the next one
* https://towardsdatascience.com/sequence-2-sequence-model-with-attention-mechanism-9e9ca2a613a
* https://towardsdatascience.com/attention-and-its-different-forms-7fc3674d14dc
* https://distill.pub/2016/augmented-rnns/ 

## Code Implementation

* https://www.analyticsvidhya.com/blog/2019/11/comprehensive-guide-attention-mechanism-deep-learning/ --> Basic Level
* https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html ---> Implementation from Scratch in Pytorch

# Transformers : Attention is all you need

So finally we have reached the end of the learning curve and are about to start learning the technology that changed NLP completely and are the reasons for the state of the art NLP techniques .Transformers were introduced in the paper Attention is all you need by Google. If you have understood the Attention models,this will be very easy , Here is transformers fully explained:

* http://jalammar.github.io/illustrated-transformer/

## Code Implementation

* http://nlp.seas.harvard.edu/2018/04/03/attention.html ---> This presents the code implementation of the architecture presented in the paper by Google

# BERT and Its Implementation on this Competition

As Promised I am back with Resiurces , to understand about BERT architecture , please follow the contents in the given order :-

* http://jalammar.github.io/illustrated-bert/ ---> In Depth Understanding of BERT

After going through the post Above , I guess you must have understood how transformer architecture have been utilized by the current SOTA models . Now these architectures can be used in two ways :<br><br>
1) We can use the model for prediction on our problems using the pretrained weights without fine-tuning or training the model for our sepcific tasks
* EG: http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/ ---> Using Pre-trained BERT without Tuning

2) We can fine-tune or train these transformer models for our task by tweaking the already pre-trained weights and training on a much smaller dataset
* EG:* https://www.youtube.com/watch?v=hinZO--TEk4&t=2933s ---> Tuning BERT For your TASK

We will be using the first example as a base for our implementation of BERT model using Hugging Face and KERAS , but contrary to first example we will also Fine-Tune our model for our task

Acknowledgements : https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras


Steps Involved :
* Data Preparation : Tokenization and encoding of data
* Configuring TPU's 
* Building a Function for Model Training and adding an output layer for classification
* Train the model and get the results


```python
# Loading Dependencies
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from kaggle_datasets import KaggleDatasets
import transformers

from tokenizers import BertWordPieceTokenizer
```


```python
# LOADING THE DATA

train1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
```

Encoder FOr DATA for understanding waht encode batch does read documentation of hugging face tokenizer :
https://huggingface.co/transformers/main_classes/tokenizer.html here


```python
def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):
    """
    Encoder for encoding the text into sequence of integers for BERT Input
    """
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(max_length=maxlen)
    all_ids = []
    
    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    
    return np.array(all_ids)
```


```python
#IMP DATA FOR CONFIG

AUTO = tf.data.experimental.AUTOTUNE


# Configuration
EPOCHS = 3
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
MAX_LEN = 192
```

## Tokenization

For understanding please refer to hugging face documentation again


```python
# First load the real tokenizer
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
# Save the loaded tokenizer locally
tokenizer.save_pretrained('.')
# Reload it with the huggingface tokenizers library
fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)
fast_tokenizer
```


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=995526.0, style=ProgressStyle(descripti…


    





    Tokenizer(vocabulary_size=119547, model=BertWordPiece, add_special_tokens=True, unk_token=[UNK], sep_token=[SEP], cls_token=[CLS], clean_text=True, handle_chinese_chars=True, strip_accents=True, lowercase=False, wordpieces_prefix=##)




```python
x_train = fast_encode(train1.comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)
x_valid = fast_encode(valid.comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)
x_test = fast_encode(test.content.astype(str), fast_tokenizer, maxlen=MAX_LEN)

y_train = train1.toxic.values
y_valid = valid.toxic.values
```

    100%|██████████| 874/874 [00:35<00:00, 24.35it/s]
    100%|██████████| 32/32 [00:01<00:00, 20.87it/s]
    100%|██████████| 250/250 [00:11<00:00, 22.06it/s]



```python
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_train, y_train))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_valid, y_valid))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(x_test)
    .batch(BATCH_SIZE)
)
```


```python
def build_model(transformer, max_len=512):
    """
    function for training the BERT model
    """
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(cls_token)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
```

## Starting Training

If you want to use any another model just replace the model name in transformers._____ and use accordingly


```python
%%time
with strategy.scope():
    transformer_layer = (
        transformers.TFDistilBertModel
        .from_pretrained('distilbert-base-multilingual-cased')
    )
    model = build_model(transformer_layer, max_len=MAX_LEN)
model.summary()
```


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=618.0, style=ProgressStyle(description_…


    



    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=910749124.0, style=ProgressStyle(descri…


    
    Model: "model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_word_ids (InputLayer)  [(None, 192)]             0         
    _________________________________________________________________
    tf_distil_bert_model (TFDist ((None, 192, 768),)       134734080 
    _________________________________________________________________
    tf_op_layer_strided_slice (T [(None, 768)]             0         
    _________________________________________________________________
    dense (Dense)                (None, 1)                 769       
    =================================================================
    Total params: 134,734,849
    Trainable params: 134,734,849
    Non-trainable params: 0
    _________________________________________________________________
    CPU times: user 34.4 s, sys: 13.3 s, total: 47.7 s
    Wall time: 50.8 s



```python
n_steps = x_train.shape[0] // BATCH_SIZE
train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=EPOCHS
)
```

    Train for 1746 steps, validate for 63 steps
    Epoch 1/3
    1746/1746 [==============================] - 255s 146ms/step - loss: 0.1221 - accuracy: 0.9517 - val_loss: 0.4484 - val_accuracy: 0.8479
    Epoch 2/3
    1746/1746 [==============================] - 198s 114ms/step - loss: 0.0908 - accuracy: 0.9634 - val_loss: 0.4769 - val_accuracy: 0.8491
    Epoch 3/3
    1746/1746 [==============================] - 198s 113ms/step - loss: 0.0775 - accuracy: 0.9680 - val_loss: 0.5522 - val_accuracy: 0.8500



```python
n_steps = x_valid.shape[0] // BATCH_SIZE
train_history_2 = model.fit(
    valid_dataset.repeat(),
    steps_per_epoch=n_steps,
    epochs=EPOCHS*2
)
```

    Train for 62 steps
    Epoch 1/6
    62/62 [==============================] - 18s 291ms/step - loss: 0.3244 - accuracy: 0.8613
    Epoch 2/6
    62/62 [==============================] - 25s 401ms/step - loss: 0.2354 - accuracy: 0.8955
    Epoch 3/6
    62/62 [==============================] - 7s 110ms/step - loss: 0.1718 - accuracy: 0.9252
    Epoch 4/6
    62/62 [==============================] - 7s 111ms/step - loss: 0.1210 - accuracy: 0.9492
    Epoch 5/6
    62/62 [==============================] - 7s 114ms/step - loss: 0.0798 - accuracy: 0.9686
    Epoch 6/6
    62/62 [==============================] - 7s 110ms/step - loss: 0.0765 - accuracy: 0.9696



```python
sub['toxic'] = model.predict(test_dataset, verbose=1)
sub.to_csv('submission.csv', index=False)
```

    499/499 [==============================] - 41s 82ms/step


# End Notes

This was my effort to share my learnings so that everyone can benifit from it.As this community has been very kind to me and helped me in learning all of this , I want to take this forward. I have shared all the resources I used to learn all the stuff .Join me and make these NLP competitions your first ,without being overwhelmed by the shear number of techniques used . It took me 10 days to learn all of this , you can learn it at your pace and dont give in , at the end of all this you will be a different person and it will all be worth it.


### I am attaching more resources if you want NLP end to end:

1) Books

* https://d2l.ai/
* Jason Brownlee's Books

2) Courses

* https://www.coursera.org/learn/nlp-sequence-models/home/welcome
* Fast.ai NLP Course

3) Blogs and websites

* Machine Learning Mastery
* https://distill.pub/
* http://jalammar.github.io/

**<span style="color:Red">This is subtle effort of contributing towards the community, if it helped you in any way please show a token of love by upvoting**
