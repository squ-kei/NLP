# Metrics
When use F1score for a multi-label classification problem, *set num_classes=1*
```python
tfa.metrics.F1Score(num_classes=1, average='micro',threshold=0.5)
```
# Tokenization
*tokenizer.word_index* is computed the same way no matter how many most frequent words you will use later(governed by num_words parameter).     
So when you call any transformative method - Tokenizer will use only the most common words you desired and at the same time, it will keep the counter of all words - even when it's obvious that it will not use it later.    
For example
```python
from tensorflow.keras.preprocessing.text import Tokenizer
vocab_size = 20000

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(all_text_sequences)
tokenizer.texts_to_sequences(x_train)
```
This code will keep the 20000 most frequent words when transform *x_train* into integer values, if oov_token is not set, all words beyond the 20000 vocabulary will be omitted. While *tokenizer.word_index* will keep the full vocabulary of *all_text_sequences*
