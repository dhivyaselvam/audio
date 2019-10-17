#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re


# In[2]:


df=pd.read_csv("C:\\Users\\Dhivya\\train.csv")


# In[3]:


df.head()


# In[4]:


df['question1']
input=df['question1'][2]


# In[5]:


input


# In[6]:


# Preprocessing the data
input = re.sub(r'\[[0-9]*\]',' ',input)
input = re.sub(r'\s+',' ',input)
clean_text = input.lower()
clean_text = re.sub(r'\W',' ',clean_text)
clean_text = re.sub(r'\d',' ',clean_text)
clean_text = re.sub(r'\s+',' ',clean_text)


# In[7]:


stop_words = nltk.corpus.stopwords.words('english')


# In[8]:


sentences = nltk.sent_tokenize(input)
sentences


# In[9]:


input1=input.lower()


# In[10]:


input1.split()


# In[ ]:


import win32com.client 
import speech_recognition as sr
i="yes"
a=[]
while (i=="yes"): 
    speaker = win32com.client.Dispatch("SAPI.SpVoice") 
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("hey, Tell me about your project? ")
        speaker.Speak("hey, Tell me about your phone ?  ")
        print("Listining now ..... ")
        #print("Speak now :")
        #speaker.Speak("Speak now :")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            print("You said : {}".format(text))
            a.append(text)
        except:
            print("Sorry could not recognize what you said")
    #i=1
    r1 = sr.Recognizer() 
    
    #i=="no"
    with sr.Microphone() as source:
        print("should we continue ?")
        speaker.Speak("should we continue ? ")
        print("Listining now ..... ")
        audio1 = r1.listen(source)
        try:
            i = r1.recognize_google(audio1)
            print("You said : {}".format(i))
            #a.append(text)
        except:
            print("Sorry could not recognize what you said")
            

print(a)
y=model.predict(vect.transform(a))
print(y)
r=0
for t in y:
    r+=1
    if t==0:
        speaker.Speak("your Feedback for phone {} is recognized as Bad Review!".format(r))
    else:
        speaker.Speak("your Feedback for phone {} is recognized as Good Review!".format(r))
# if y[0]==0:
#     speaker.Speak("your Feedback is recognized as Bad Review!")
# else:
#     speaker.Speak("your Feedback is recognized as Good Review!")


# In[ ]:





# In[12]:


df.isnull().sum()


# In[13]:


df.dropna(axis=0,inplace=True)


# In[14]:


cos_score=[]
for i in df['question1']:
    text=i
    tfidf1.text=tfidf[i]
    for j in df['question2']:
        tfidf2.text=tfidf[j]
        cos=cos(tfidf1,tfidf2)
    cos_score.append(i,j,cos)


# In[16]:


from numpy import dot
from numpy.linalg import norm


# In[15]:


from sklearn.feature_extraction.text import CountVectorizer

# Fit the CountVectorizer to the training data
vect = CountVectorizer().fit(df['question1'].sample(1000))
#vect1= CountVectorizer().fit('input')


# In[17]:


# transform the documents in the training data to a document-term matrix
X_train_vectorized = vect.transform(df['question1'].sample(1000))

X_train_vectorized


# In[18]:


X_train_vectorized.toarray()


# In[19]:


arry = X_train_vectorized.toarray()
row1 =  arry[0]
cosinelist =[]

#cos_sim = dot(a, b)/(norm(a)*norm(b)
for i in arry:
    #count=0
    cos_sim = dot(row1,i)/(norm(row1)*norm(i))
    #print(product)
    #break
    cosinelist.append(cos_sim)


# In[20]:


cosinelist


# In[21]:


df=df[['question1','question2']]


# In[22]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidfconverter=TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7, stop_words=stopwords.words("english"))

tfidfconverter.fit(df['question1'].sample(1000))
tfidf_text1= tfidfconverter.transform(df['question1'].sample(1000))
tfidf_text2= tfidfconverter.transform(df['question2'].sample(1000))


# In[23]:


tfidf_text1=tfidf_text1.toarray()
tfidf_text1


# In[24]:


tfidf_text2=tfidf_text2.toarray()
tfidf_text2


# In[25]:


finalist=[]
count=0
for i in tfidf_text1:
    count=count+1
    cos_sim=[]
    for j in tfidf_text2:
        cos_sim.append(dot(i,j)/(norm(i)*norm(j)))
    finalist.append([count,cos_sim])


# In[26]:


finalist=pd.DataFrame(finalist)


# In[27]:


newfile=pd.DataFrame(finalist[1].values.tolist())
newfile


# In[35]:


test1=[]
for i in range(0,newfile.shape[0]):
    max1=newfile[i].values.max()
    list1=list(newfile[i].values)
    list1.remove(max1)
    test1.append(argmax())
    


# In[ ]:


for i in test1:
    print(df[i+1][i+2])


# In[28]:


newfile.values.argmax(axis=1)


# In[29]:


for i in newfile.columns:
    finalist[i]=newfile[i]


# In[30]:


finalist.fillna(0,inplace=True)


# In[31]:


finalist.values.argmax(axis=1)


# In[ ]:




