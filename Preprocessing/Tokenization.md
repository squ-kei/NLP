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
