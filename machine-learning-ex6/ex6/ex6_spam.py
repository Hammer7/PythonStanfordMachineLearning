import os
import numpy as np
import re
import string
from nltk.stem import PorterStemmer
from scipy.io import loadmat
from svm import SVM

#  Exercise 6 | Spam Classification with SVMs
scriptdir = os.path.dirname(os.path.realpath(__file__))

def linearKernel(x1, x2):
    return x1.T @ x2


def readFile(filename):
    with open(scriptdir + '//' + filename, 'r') as f:
        file_contents = f.read()
    return file_contents

def getVocabList():
    vocab = np.genfromtxt(scriptdir + '//vocab.txt', dtype='str', delimiter='\t')
    vocabDict = { word:int(index) for index,word in vocab }
    return vocabDict

def processEmail(email_contents):
    # ========================== Preprocess Email ===========================

    # Find the Headers ( \n\n and remove )
    # Uncomment the following lines if you are working with raw emails with the full headers

    # hdrstart = strfind(email_contents, ([char(10) char(10)]));
    # email_contents = email_contents(hdrstart(1):end);
    # Lower case
    email_contents = email_contents.lower()

    # Strip all HTML
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)

    # Handle Numbers
    # Look for one or more characters between 0-9
    email_contents = re.sub('[0-9]+', 'number', email_contents)

    # Handle URLS
    # Look for strings starting with http:// or https://
    email_contents = re.sub(r'(http|https)://[^\s]*', 'httpaddr', email_contents)

    # Handle Email Addresses
    # Look for strings with @ in the middle
    email_contents = re.sub(r'[^\s]+@[^\s]+', 'emailaddr', email_contents)

    # Handle $ sign
    email_contents = re.sub('[$]+', 'dollar', email_contents)

    # ========================== Tokenize Email ===========================

    # Output the email to screen as well
    print('\n==== Processed Email ====\n')

    # Process file
    # Tokenize and also get rid of any punctuation
    translator = str.maketrans(string.punctuation+'\n\r',(len(string.punctuation)+2) * ' ')
    words = email_contents.translate(translator).split()

        # Init return value
    word_indices = np.zeros(len(words), dtype=int) #max length


    stemmer = PorterStemmer()
    count = 0
    l = 0
    for word in words:   
        # Remove any non alphanumeric characters
        word = re.sub('[^a-zA-Z0-9]', '', word)
        # Stem the word  (the porterStemmer sometimes has issues, so we use a try catch block)
        try:
            word = stemmer.stem(word)
        except:
             word = ''; continue

        # Skip the word if it is too short
        if not word: #empty
            continue
        index = vocabDict.get(word)
        if index is not None:
            word_indices[count] = index
            count += 1
        # Print to screen, ensuring that the output lines are not too long
        if (l + len(word) + 1) > 78:
            print('')
            l = 0
        print(word, end=' ')
        l = l + len(word) + 1
    # Print footer
    print('\n\n=========================\n')
    return np.trim_zeros(word_indices, 'b') #remove zeros from back

def emailFeatures(word_indices):
    # Total number of words in the dictionary
    n = 1899
    x = np.zeros(n)
    x[word_indices] = 1
    return x

## ==================== Part 1: Email Preprocessing ====================
print('\nPreprocessing sample email (emailSample1.txt)')

vocabDict = getVocabList()
print(vocabDict)
# Extract Features
file_contents = readFile('emailSample1.txt')

word_indices  = processEmail(file_contents)

# Print Stats
print('Word Indices: ')
print( word_indices)
print('')

input('Program paused. Press enter to continue.\n')

## ==================== Part 2: Feature Extraction ====================

print('\nExtracting features from sample email (emailSample1.txt)')

# Extract Features
file_contents = readFile('emailSample1.txt')
word_indices  = processEmail(file_contents)
features      = emailFeatures(word_indices)

# Print Stats
print(f'Length of feature vector: {features.size}')
print(f'Number of non-zero entries: {sum(features)}')

input('Program paused. Press enter to continue.\n')

## =========== Part 3: Train Linear SVM for Spam Classification ========

# Load the Spam Email dataset
# You will have X, y in your environment
data = loadmat(scriptdir + '//spamTrain.mat')
X = data['X']
y = data['y'].ravel()
print('\nTraining Linear SVM (Spam Classification)')
print('(this may take 1 to 2 minutes) ...')

import cProfile, pstats, io
pr = cProfile.Profile()
pr.enable()
# ... do something ...
C = 0.1
model = SVM()
model.svmTrain(X, y.astype(float), C, linearKernel)

p = model.svmPredict(X)

pr.disable()
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())

print(f'Training Accuracy: {np.mean(p == y) * 100}')

## =================== Part 4: Test Spam Classification ================

# Load the test dataset
# You will have Xtest, ytest in your environment
data = loadmat(scriptdir + '//spamTest.mat')
Xtest = data['Xtest']
ytest = data['ytest'].ravel()

print('\nEvaluating the trained Linear SVM on a test set ...')

p = model.svmPredict(Xtest)

input(f'Test Accuracy: {np.mean(p == ytest) * 100}\n')

## ================= Part 5: Top Predictors of Spam ====================

# Sort the weights and obtain the vocabulary list
idx = np.argsort(model.w)[::-1]
vocabDict = getVocabList()
print('\nTop predictors of spam: ')
for i in range(15):
    for word, index in vocabDict.items():
        if index == idx[i]:
            print(f'{word:<15} {model.w[idx[i]]}')
            break

print('\n\n')
input('\nProgram paused. Press enter to continue.\n')


## =================== Part 6: Try Your Own Emails =====================

# Set the file to be read in (change this to spamSample2.txt,
# emailSample1.txt or emailSample2.txt to see different predictions on
# different emails types). Try your own emails as well!
samples = ['spamSample1.txt', 'spamSample2.txt', 'emailSample1.txt', 'emailSample2.txt']
for filename in samples:
    # Read and predict
    file_contents = readFile(filename)
    word_indices  = processEmail(file_contents)
    x             = emailFeatures(word_indices)
    p = model.svmPredict(x.reshape(1,-1))

    print(f'\nProcessed {filename}\n\nSpam Classification: {p}')
    print('(1 indicates spam, 0 indicates not spam)\n')
    input('\nnProgram paused. Press enter to continue\n')