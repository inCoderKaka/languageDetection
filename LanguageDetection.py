#!/usr/bin/env python
# coding: utf-8

# In[1]:


import speech_recognition as sr
r = sr.Recognizer()

import glob

import fasttext
from pycountry import languages


# In[2]:


PRETRAINED_MODEL_PATH = 'lid.176.ftz'
model = fasttext.load_model(PRETRAINED_MODEL_PATH)


# In[3]:


def audioToString(path):
    audioFile = sr.AudioFile(path)
    with audioFile as source:
        audioData = r.record(source)
    return r.recognize_google(audioData)


# In[4]:


def finalPrediction(path):
    sentences = audioToString(path)
    predictions = model.predict(sentences)
    lang_name = languages.get(alpha_2=predictions[0][0][9:]).name
    if lang_name != 'English':
        lang_name = 'Hindi'
    return lang_name


# In[5]:


#For Single File :
finalPrediction('C:\\Users\\kakar\\Documents\\DecibellCache\\New Test\\harvard.wav')


# In[7]:


#For multiple Files :
files = glob.glob('C:\\Users\\kakar\\Documents\\DecibellCache\\New Test\\*.wav')
for file in files:
    print(file.split('\\')[-1],'  -->\t',finalPrediction(file))


# In[ ]:




