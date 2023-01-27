# # ASCII to char and vice cersa
import sys

print("ASCII to char and vice versa")
char = 'A'
print(ord(char))

ascii = 65
print(chr(ascii))


# One-Hot Encoding
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

print("\nOne-Hot Encoding")
docs = "the bird and the bee"

# memisah kalimat menjadi token
split_docs = docs.split(" ")
data = [doc.split(" ") for doc in split_docs]
values = array(data).ravel()

# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)


# CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

print("\ncountVectorizer")
text = ["everybody love nlp", "nlp is so cool", 
"nlp is all about helping machines process language", 
"this tutorial is on basic nlp technique"]

vectorizer = CountVectorizer()

# tokenisasi dan membuat vocab
vectorizer.fit(text)
print(vectorizer.vocabulary_)


# encode dokumen
vector = vectorizer.transform(text)

# hasil encode vektor
print(vector.shape) 
print()
print(vector.toarray())

# #3.TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer

print("\nTF-IDF")
text1 = ['i love nlp', "nlp is so cool", 
"nlp is all about helping machines process language", 
"this tutorial is on basic nlp technique"]

tf = TfidfVectorizer()
txt_fitted = tf.fit(text1)
txt_transformed = txt_fitted.transform(text1)

idf = tf.idf_
print(dict(zip(txt_fitted.get_feature_names(), idf)))
