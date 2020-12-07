# Tokenization strategies    
Basically we have 4 different strategies as listed below:   
1.into sentences    
2.into words   
3.into characters   
4.into subwords     
**Different models/tasks require different kind of input.** Some need list of sentences like w2v. Some need list of lists of words. Some need subwords tokens.      
Check carefully when use pretrained models as they generally require different input format.        

There are a lot of tools that can be used to tokenize text. Some common ones are NLTK, GENSIM, SPACY, TEXTBLOB, keras.preprocessing.text.Tokenizer 
(these mainly use space or rule based tokenization) or subwords tokenization with [transformers package](https://huggingface.co/transformers/tokenizer_summary.html) provided by
hugging face. 

Why we need tokenization?
Because most NLP models process input text as a sequence of words or tokens. Tokenization is responsible for decomposing whole chunk of text into smaller units(tokens).
After that, usually each token will be converted into a numerical representation(integer index, one-hot enocoding, embedding,etc.) such that it can be recognized and understood 
by machines.

This document will mainly focus on keras and transformers implementation as they are the most common tokenization tools for deep learning nlp models.

## Tokenize into sentences    
This is generally require for word2vec. Below is an example using NLTK:
```python
# load data
filename = 'textfile.txt'
with open(filename, 'rt') as file:
     text = file.read()
# split into sentences
from nltk import sent_tokenize
sentences = sent_tokenize(text)
print(sentences[0])
```

## Tokenize into words
Several different packages will do the work including NLTK, gensim, spaCy, and keras.preprocessing.text.tokenizer.
Keras tokenizer is






text--->sequence of integer --> embedded sequence
     |->dict(word:index)-->match pretrained embeddings-->dict(index:embedding vector)
     
fit_on_sequence?
sequence/text to matrix? this is mainly for count based methods, only have 4 modes:"binary", "count", "tfidf", "freq",
matrix is m*n where m is number of sequences(instances), n is the length of vocabulary 

```python
t  = Tokenizer()
fit_text = "The earth is an awesome place live"
t.fit_on_texts(fit_text)
test_text = "The earth is an great place live"
sequences = t.texts_to_sequences(test_text)

print("sequences : ",sequences,'\n')

print("word_index : ",t.word_index)
#[] specifies : 1. space b/w the words in the test_text    2. letters that have not occured in fit_text

Output :

       sequences :  [[3], [4], [1], [], [1], [2], [8], [3], [4], [], [5], [6], [], [2], [9], [], [], [8], [1], [2], [3], [], [13], [7], [2], [14], [1], [], [7], [5], [15], [1]] 

       word_index :  {'e': 1, 'a': 2, 't': 3, 'h': 4, 'i': 5, 's': 6, 'l': 7, 'r': 8, 'n': 9, 'w': 10, 'o': 11, 'm': 12, 'p': 13, 'c': 14, 'v': 15}
```

```python
t  = Tokenizer()
fit_text = ["The earth is an awesome place live"]
t.fit_on_texts(fit_text)

#fit_on_texts fits on sentences when list of sentences is passed to fit_on_texts() function. 
#ie - fit_on_texts( [ sent1, sent2, sent3,....sentN ] )

#Similarly, list of sentences/single sentence in a list must be passed into texts_to_sequences.
test_text1 = "The earth is an great place live"
test_text2 = "The is my program"
sequences = t.texts_to_sequences([test_text1, test_text2])

print('sequences : ',sequences,'\n')

print('word_index : ',t.word_index)
#texts_to_sequences() returns list of list. ie - [ [] ]

Output:

        sequences :  [[1, 2, 3, 4, 6, 7], [1, 3]] 

        word_index :  {'the': 1, 'earth': 2, 'is': 3, 'an': 4, 'awesome': 5, 'place': 6, 'live': 7}
```
**it will ignore unknown words if not set oov_token**      
keras tokenizer index start from 1 not 0, 0 is for padding.



What does a pretrained subword tokenizer does when applied to new data?
It will save the vocabulary and merge rule when training. While applied to new data, it does the same merge based upon the merge rule(vocabulary and corresponding count/probability obtained from training).
