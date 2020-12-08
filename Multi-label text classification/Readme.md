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

# A thing with tf.data.Dataset
```python
model.fit(train_dataset.shuffle(1000).batch(16), batch_size=16, epochs=3,validation_data=val_dataset.batch(16))
```
With the above code, two things need to be careful.     
First, if dataset has *batch()* method while *fit* also has *batch_size* parameter, they will be multiplied to produce the final batch size. 16*16=256 for the above example.     
Second, basically you need to have the validation set with the same batch size or else it will raise a *ValueError: logits and labels must have the same shape ((256, 6) vs (6, 1))*. So for the above example, change it to
```python
model.fit(train_dataset.shuffle(1000).batch(16), batch_size=16, epochs=3,validation_data=val_dataset.batch(256))
```
or
```python
model.fit(train_dataset.shuffle(1000).batch(16), epochs=3,validation_data=val_dataset.batch(16))
```
