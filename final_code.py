# %% [markdown]
# # Importing Libraries

# %%
import nltk
from nltk.corpus import stopwords
from nltk.corpus import treebank
import string
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd 
from nltk.tokenize import word_tokenize

# %% [markdown]
# # Operating System Textbook
# 

# %%
nltk.download('stopwords')

# %%
nltk.download('punkt')

# %%
nltk.download('averaged_perceptron_tagger')

# %%
file = open(r"updated_book_text.txt",encoding='utf-8')
wordslist = file.read().splitlines() # to escape \n occurence
wordslist = [i for i in wordslist if i!='']
text = " ".join(wordslist)

# %%
len(text)

# %% [markdown]
# # Preprocessing
# 

# %%
import re

#Creating a string which has all the punctuations to be removed
punctuations = '''!()-[]{};:'"\,<>./‘’?“”@#$%^&*_~'''
cleantext = ""
for char in text:
    if char not in punctuations:
        cleantext = cleantext + char
        
#Converting the text into lower case         
cleantext = cleantext.lower()
cleantext = re.sub(r'[ ]?this page intentionally left blank[ ]?','',cleantext)
cleantext = re.sub(r'chapter[ ]?[0-9]+','',cleantext)
cleantext = re.sub(r'figure[ ]?[0-9]+','',cleantext)
cleantext = re.sub(r'preface','',cleantext)
cleantext = re.sub(r'^m{0,3}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$','',cleantext)
cleantext = re.sub(r'[ ]e[ ]','',cleantext)
cleantext = re.sub(r'[0-9]+','',cleantext)
cleantext = re.sub(r' www[a-z]+ ','',cleantext)
cleantext = re.sub(r' [^aci] ','',cleantext)


# %%
# cleantext

# %%
len(cleantext)

# %%
wordcloud = WordCloud(width = 800, height = 600, 
                background_color ='white', 
                min_font_size = 10,stopwords = {},colormap='winter').generate(cleantext) 

plt.figure(figsize = (12,8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 

# %%
tokens = word_tokenize(cleantext)
freq = nltk.FreqDist(tokens)
plt.figure(figsize=(12,5))
freq.plot(40, cumulative=False)

# %%
tokens = word_tokenize(cleantext)
# tokens

# %%
len(tokens)

# %% [markdown]
# ### Removing stopwords and tokenising

# %%
# Removing stopwords and storing it into finaltext
stop_words = set(stopwords.words('english'))
tokens = word_tokenize(cleantext)
tokens_final = [i for i in tokens if not i in stop_words] # tokenising with removing stopwords
finaltext = "  "
finaltext = finaltext.join(tokens_final)

# %%
len(finaltext)

# %%
# Word cloud after removing stopwords
wordcloud = WordCloud(width = 800, height = 600, 
                background_color ='white', 
                min_font_size = 10,stopwords = {},colormap='winter').generate(finaltext) 

plt.figure(figsize = (12,8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 

# %% [markdown]
# ### Frequency distribution of tokens

# %%
tokens = word_tokenize(finaltext)
tokens = [i for i in tokens if not i in stop_words]
freq = nltk.FreqDist(tokens)
plt.figure(figsize=(12,5))
freq.plot(40, cumulative=False)

# %%
tagged = nltk.pos_tag(tokens) 
# tagged

# %%
from collections import Counter
counts = Counter( tag for word,  tag in tagged)
print(counts)

# %%
freq_tags = nltk.FreqDist(counts)
plt.figure(figsize=(12,5))
freq_tags.plot(40, cumulative=False)

# %% [markdown]
# ## For word length vs Frequency distribution

# %%
import numpy as np
bin_size=np.linspace(0,25)

# %%
#Finding Wordlength and storing it as a list
wordLength = [len(r) for r in tokens]

#Plotting histogram of Word length vs Frequency
plt.hist(wordLength, bins=bin_size)
plt.xlabel('word length')
plt.ylabel('word length Frequency')
plt.title('Frequency Distribution for the book Operating System')
plt.show()



