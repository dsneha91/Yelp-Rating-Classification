import sys

import pickle
import gzip

def save_zipped_pickle(obj, filename, protocol=-1):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object

test = {1:'qwe'}
save_zipped_pickle(test, 'w2vmodel-training1-300d-iter5-hs1-neg0.pkl.gz')

#if len(sys.argv) < 3:
#    exit("format: python deepir.py trainset_label testset_label")

trainset_label = 'training1'
#testset_label = sys.argv[2]

import re
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

from zipfile import ZipFile
import json

def YelpReviews(label):
    with ZipFile("data/yelp_%s_set.zip"%label, 'r') as zf:
        with zf.open("yelp_%s_set/yelp_%s_set_review.json"%(label,label)) as f:
            for line in f:
                rev = json.loads(line.decode())
                yield {'y':rev['stars'],'x':[clean(s).split() for s in sentences(rev['text'])]}


# Now, since the files are small we'll just read everything into in-memory lists.  It takes a minute ...

# In[10]:

revtrain = list(YelpReviews(trainset_label))
print(len(revtrain), "training reviews")

## and shuffle just in case they are ordered
import numpy as np
np.random.shuffle(revtrain)


# Finally, write a function to generate sentences -- ordered lists of words -- from reviews that have certain star ratings

# In[11]:

def Sentences(reviews):
    for r in reviews:
        for s in r['x']:
            yield s


# ### Word2Vec modeling

# We fit out-of-the-box Word2Vec

# In[12]:

from gensim.models import Word2Vec
import multiprocessing

## create a w2v learner 
basemodel = Word2Vec(
    workers=multiprocessing.cpu_count(), # use your cores
    size=300,
    iter=5, # iter = sweeps of SGD through the data; more is better
    hs=1, negative=0 # we only have scoring for the hierarchical softmax setup
    )
print(basemodel)


# Build vocab from all sentences (you could also pre-train the base model from a neutral or un-labeled vocabulary)

# In[13]:

basemodel.build_vocab(Sentences(revtrain))

# Now, we will _deep_ copy each base model and do star-specific training. This is where the big computations happen...

# In[14]:

slist = list(Sentences(revtrain))
basemodel.train(  slist, total_examples=len(slist) )

save_zipped_pickle(basemodel, 'w2vmodel-training1-300d-iter5-hs1-neg0.pkl.gz')
