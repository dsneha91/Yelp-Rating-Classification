
# coding: utf-8

# ## Deep Inverse Regression with Yelp reviews
# 
# In this note we'll use [gensim](http://radimrehurek.com/gensim/) to turn the Word2Vec machinery into a document classifier, as in [Document Classification by Inversion of Distributed Language Representations](http://arxiv.org/pdf/1504.07295v3) from ACL 2015.

# First, we define a super simple parser

# In[3]:

import sys
import re
from zipfile import ZipFile
import json
import numpy as np
from gensim.models import Word2Vec
import multiprocessing
from copy import deepcopy
import pandas as pd # for quick summing within doc

if len(sys.argv) < 3:
    exit("format: python deepir.py trainset_label testset_label")

trainset_label = sys.argv[1]
testset_label = sys.argv[2]

contractions = re.compile(r"'|-|\"")
# all non alphanumeric
symbols = re.compile(r'(\W+)', re.U)
# single character removal
singles = re.compile(r'(\s\S\s)', re.I|re.U)
# separators (any whitespace)
seps = re.compile(r'\s+')

# cleaner (order matters)
def clean(text): 
    text = text.lower()
    text = contractions.sub('', text)
    text = symbols.sub(r' \1 ', text)
    text = singles.sub(' ', text)
    text = seps.sub(' ', text)
    return text

# sentence splitter
alteos = re.compile(r'([!\?])')
def sentences(l):
    l = alteos.sub(r' \1 .', l).rstrip("(\.)*\n")
    return l.split(".")


# And put everything together in a review generator that provides tokenized sentences and the number of stars for every review.

# In[6]:

def YelpReviews(label):
    with ZipFile("yelp_%s_set.zip"%label, 'r') as zf:
        with zf.open("yelp_%s_set/yelp_%s_set_review.json"%(label,label)) as f:
            for line in f:
                rev = json.loads(line.decode())
                yield {'y':rev['stars'],'x':[clean(s).split() for s in sentences(rev['text'])]}

# For example:

# In[9]:

#YelpReviews(testset_label).next()


# Now, since the files are small we'll just read everything into in-memory lists.  It takes a minute ...

# In[10]:

revtrain = list(YelpReviews(trainset_label))
print(len(revtrain), "training reviews")

## and shuffle just in case they are ordered
np.random.shuffle(revtrain)


# Finally, write a function to generate sentences -- ordered lists of words -- from reviews that have certain star ratings

# In[11]:

def StarSentences(reviews, stars=[1,2,3,4,5]):
    for r in reviews:
        if r['y'] in stars:
            for s in r['x']:
                yield s


# ### Word2Vec modeling

# We fit out-of-the-box Word2Vec

# In[12]:


## create a w2v learner 
basemodel = Word2Vec(
    workers=multiprocessing.cpu_count(), # use your cores
    iter=3, # iter = sweeps of SGD through the data; more is better
    hs=1, negative=0 # we only have scoring for the hierarchical softmax setup
    )
print(basemodel)


# Build vocab from all sentences (you could also pre-train the base model from a neutral or un-labeled vocabulary)

# In[13]:

basemodel.build_vocab(StarSentences(revtrain))  


# Now, we will _deep_ copy each base model and do star-specific training. This is where the big computations happen...

# In[14]:

starmodels = [deepcopy(basemodel) for i in range(5)]
for i in range(5):
    slist = list(StarSentences(revtrain, [i+1]))
    print(i+1, "stars (", len(slist), ")")
    starmodels[i].train(  slist, total_examples=len(slist) )
    


# ### Inversion of the distributed representations
# 
# At this point, we have 5 different word2vec language representations.  Each 'model' has been trained conditional (i.e., limited to) text from a specific star rating.  We will apply Bayes rule to go from _p(text|stars)_ to _p(stars|text)_.

# For any new sentence we can obtain its _likelihood_ (lhd; actually, the composite likelihood approximation; see the paper) using the [score](https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec.score) function in the `word2vec` class.  We get the likelihood for each sentence in the first test review, then convert to a probability over star ratings. Every sentence in the review is evaluated separately and the final star rating of the review is an average vote of all the sentences. This is all in the following handy wrapper.

# In[49]:

"""
docprob takes two lists
* docs: a list of documents, each of which is a list of sentences
* models: the candidate word2vec models (each potential class)

it returns the array of class probabilities.  Everything is done in-memory.
"""


def docprob(docs, mods):
    # score() takes a list [s] of sentences here; could also be a sentence generator
    sentlist = [s for d in docs for s in d]
    # the log likelihood of each sentence in this review under each w2v representation
    llhd = np.array( [ m.score(sentlist, len(sentlist)) for m in mods ] )
    # now exponentiate to get likelihoods, 
    lhd = np.exp(llhd - llhd.max(axis=0)) # subtract row max to avoid numeric overload
    # normalize across models (stars) to get sentence-star probabilities
    prob = pd.DataFrame( (lhd/lhd.sum(axis=0)).transpose() )
    # and finally average the sentence probabilities to get the review probability
    prob["doc"] = [i for i,d in enumerate(docs) for s in d]
    prob = prob.groupby("doc").mean()
    return prob


# #### Test set example
# 
# As an example, we apply the inversion on the full test set.  

# In[42]:

# read in the test set
revtest = list(YelpReviews(testset_label))


# In[50]:

### get the probs (note we give docprob a list of lists of words, plus the models)
probs = docprob( [r['x'] for r in revtest], starmodels )
##
##
### In[13]:
##
##get_ipython().magic(u'matplotlib inline')
##
##probpos = pd.DataFrame({"out-of-sample prob positive":probs[[3,4]].sum(axis=1), 
##                        "true stars":[r['y'] for r in revtest]})
##probpos.boxplot("out-of-sample prob positive",by="true stars", figsize=(12,5))
##
##
### In[18]:
##
##get_ipython().magic(u'matplotlib inline')
##
##probpos = pd.DataFrame({"out-of-sample prob positive":probs[[3,4]].sum(axis=1), 
##                        "true stars":[r['y'] for r in revtest]})
##probpos.boxplot("out-of-sample prob positive",by="true stars", figsize=(12,5))
##

# In[68]:

#probs.shape


# In[74]:

true_stars = [r['y'] for r in revtest]
predicted_stars = [np.argmax(row)+1 for row in probs.transpose()]


# In[77]:

num_predicted_correctly = 0
for i in range(len(true_stars)):
    if true_stars[i] == predicted_stars[i]:
        num_predicted_correctly += 1
        
print("accuracy: ",1.0*num_predicted_correctly/len(probs))

