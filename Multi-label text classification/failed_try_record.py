class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, labels=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            labels: (Optional) [string]. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels

######################################
def create_examples(df, labels_available=True):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, row) in enumerate(df.values):
        guid = row[0]
        text_a = row[1]
        if labels_available:
            labels = row[2:]
        else:
            labels = [0,0,0,0,0,0]
        examples.append(
            InputExample(guid=guid, text_a=text_a, labels=labels))
    return examples
######################################
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, labels, is_real_example=True):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.labels = labels
        self.is_real_example=is_real_example
        
        
trainexamples = create_examples(trainset)
valexamples = create_examples(valset)

def convert_examples_to_features(examples,  max_seq_length, tokenizer):
    features = []
    for example in examples:
        tokens_a = tokenizer(example.text_a,
                             add_special_tokens=True,
                             max_length=max_seq_length,
                             truncation=True,
                             padding=True, 
                             return_tensors='tf')
        #features.append(
        #    InputFeatures(input_ids=tokens_a.input_ids, 
        #                  attention_mask=tokens_a.attention_mask, 
        #                  token_type_ids=tokens_a.token_type_ids, 
        #                  labels=example.labels))
        features.append({"input_ids":tokens_a.input_ids,
                        "attention_mask":tokens_a.attention_mask,
                        "token_type_ids":tokens_a.token_type_ids,
                        "labels":example.labels})
    return features
    
    
trainfeatures = convert_examples_to_features(trainexamples,  max_seq_length=256, tokenizer=tokenizer)
valfeatures = convert_examples_to_features(valexamples,  max_seq_length=256, tokenizer=tokenizer)

###################################
def convert_examples_to_features2(examples,  max_seq_length, tokenizer):
    features = []
    labels = []
    for example in examples:
        tokens_a = tokenizer(example.text_a,
                             add_special_tokens=True,
                             max_length=max_seq_length,
                             truncation=True,
                             padding=True, 
                             return_tensors='tf')
        #features.append(
        #    InputFeatures(input_ids=tokens_a.input_ids, 
        #                  attention_mask=tokens_a.attention_mask, 
        #                  token_type_ids=tokens_a.token_type_ids, 
        #                  labels=example.labels))
        features.append({"input_ids":tokens_a.input_ids,
                        "attention_mask":tokens_a.attention_mask,
                        "token_type_ids":tokens_a.token_type_ids
                        })
        labels.append(example.labels)
    return features,labels
    
    
trainfeatures2, trainlabels = convert_examples_to_features2(trainexamples,  max_seq_length=256, tokenizer=tokenizer)
valfeatures2, vallabels = convert_examples_to_features2(valexamples,  max_seq_length=256, tokenizer=tokenizer)


trainlabels2 = [tf.constant(i,dtype=tf.float32) for i in trainlabels]
vallabels2 = [tf.constant(i,dtype=tf.float32) for i in vallabels]


#Try with Hugging Face's model
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)


optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss)
history = model.fit(x=trainfeatures2, y=trainlabels2,epochs=2, steps_per_epoch=115, validation_data=(valfeatures2, vallabels2))


#Define own model
class BertForMultilabelClassification(tf.keras.Model):
    def __init__(self,num_labels=6,**kwargs):
        super().__init__(**kwargs)
        self.bert = TFBertModel.from_pretrained('bert-base-uncased')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense = tf.keras.layers.Dense(num_labels,kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                                          bias_initializer=tf.zeros_initializer())
    def call(self,features):
        _,z = self.bert(input_ids=features.input_ids,
                     attention_mask=features.attention_mask,
                     token_type_ids=features.token_type_ids)
        z = self.dropout(z)
        return self.dense(z)
        
        
        
model3 = BertForMultilabelClassification()


optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
model3.compile(optimizer=optimizer, loss=loss)
history = model3.fit({
     "input_ids": trainfeatures.input_ids,
     "attention_mask": trainfeatures.attention_mask,
     "token_type_ids": trainfeatures.token_type_ids,
     }, y=trainlabels2,epochs=2, steps_per_epoch=115)
