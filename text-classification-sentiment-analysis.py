# feel free to include more imports as needed here
# these are the ones that we used for the base model
import numpy as np
import sys
from collections import Counter
import math
from itertools import chain
from random import randrange


def generate_tuples_from_file(training_file_path):
  """
  Generates tuples from file formated like:
  id\ttext\tlabel
  Parameters:
    training_file_path - str path to file to read in
  Return:
    a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
  """
  f = open(training_file_path, "r", encoding="utf8")
  listOfExamples = []
  for review in f:
    if len(review.strip()) == 0:
      continue
    dataInReview = review.split("\t")
    for i in range(len(dataInReview)):
      # remove any extraneous whitespace
      dataInReview[i] = dataInReview[i].strip()
    t = tuple(dataInReview)
    listOfExamples.append(t)
  f.close()
  return listOfExamples


def precision(gold_labels, predicted_labels):
  """
  Calculates the precision for a set of predicted labels give the gold (ground truth) labels.
  Parameters:
      gold_labels (list): a list of labels assigned by hand ("truth")
      predicted_labels (list): a corresponding list of labels predicted by the system
  Returns: double precision (a number from 0 to 1)
  -- true_pos/(true_pos+false_pos) --
  """
  truepos = sum( [ g == p and p == "1" for g,p in zip(gold_labels, predicted_labels)] )
  falsepos = sum( [ g != p and p == "1" for g,p in zip(gold_labels, predicted_labels)] )
  return truepos/(truepos+falsepos)


def recall(gold_labels, predicted_labels):
  """
  Calculates the recall for a set of predicted labels give the gold (ground truth) labels.
  Parameters:
      gold_labels (list): a list of labels assigned by hand ("truth")
      predicted_labels (list): a corresponding list of labels predicted by the system
  Returns: double recall (a number from 0 to 1)
  true_pos/ (true_pos +false_neg)
  """
  truepos = sum( [ g == p and p == "1" for g,p in zip(gold_labels, predicted_labels)] )
  falseneg = sum( [ g != p and p == "0" for g,p in zip(gold_labels, predicted_labels)] )
  return truepos/(truepos + falseneg)


def f1(gold_labels, predicted_labels):
  """
  Calculates the f1 for a set of predicted labels give the gold (ground truth) labels.
  Parameters:
      gold_labels (list): a list of labels assigned by hand ("truth")
      predicted_labels (list): a corresponding list of labels predicted by the system
  Returns: double f1 (a number from 0 to 1)
  2*prec*recall/(prec+recall)
  """
  p = precision(gold_labels, predicted_labels)
  r = recall(gold_labels, predicted_labels)
  if p==0 and r==0:
      return 0
  return  2 * p * r / (p + r)


class TextClassify:


  def __init__(self):
    pass


  def train(self, examples):
    """
    Trains the classifier based on the given examples
    Parameters:
      examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    Return: None
    """
    #print("[ DEBUG ] examples: ", examples)
    sentences = [ex[1].split() for ex in examples]
    self.vocabulary = set(chain.from_iterable(sentences))
    self.VOCAB_SIZE = len(self.vocabulary)
    #print("[ DEBUG ] vocab: ", self.vocabulary)
    #print("[ DEBUG ] vocab size: ",  self.VOCAB_SIZE)

    # store example words with labels in the format (word, label)
    classes = []
    for ex in examples:
        words = ex[1].split()
        for word in words:
            classes.append( (word, ex[2]) )
    #print("[ DEBUG ] txt,class list for train set: ", classes)

    class1, class0 = [], []

    # store number of docs for prior calculations
    self.n_zeros = len([ex for ex in examples if ex[2] == "0"])
    self.n_docs = len(examples)

    #print("[ DEBUG ] n_zeros: ", self.n_zeros)
    #print("[ DEBUG ] n_ones: ", self.n_docs - self.n_zeros)
    #print("[ DEBUG ] prior for zero: ", self.n_zeros/self.n_docs)
    #print("[ DEBUG ] prior for one: ", 1- (self.n_zeros/self.n_docs))

    # store words for likelihood calculations
    for tup in classes:
        if tup[1] == "1":
            class1.append(tup[0])
        else:
            class0.append(tup[0])

    # BoW: store counts of each word, for each class
    self.counts_zero = { txt:len([t for t in class0 if t==txt]) for txt in set(class0) }
    self.counts_one = { txt:len([t for t in class1 if t==txt]) for txt in set(class1) }

    #print("[ DEBUG ] word:count_in_class_zero: ", self.counts_zero)
    return None


  def score(self, data):

    """
    Score a given piece of text
    ( calculate e ^ (log(p(c)) + sum(log(p(w_i | c)))) )

    Parameters:
      data - str like "I loved the hotel"

    Return: dict of class: score mappings

    """

    tokens = data.split()
    tok_scores_zero, tok_scores_one = [], []

    # loop over test tokens and check if they exist in vocabulary/each class
    for tok in tokens:
        # if not in vocab, ignore the term
        if tok not in self.counts_zero.keys() and tok not in self.counts_one.keys():
            continue
        # if in vocab but exists in the BoW for only one class, put zero in
        # that class and apply smoothing to both classes
        if tok not in self.counts_zero.keys():
            tok_scores_zero.append( 1/(sum(self.counts_zero.values()) + self.VOCAB_SIZE) )
            tok_scores_one.append( (self.counts_one[tok] + 1)/(sum(self.counts_one.values()) + self.VOCAB_SIZE) )
        elif tok not in self.counts_one.keys():
            tok_scores_zero.append( (self.counts_zero[tok] + 1)/(sum(self.counts_zero.values())+ self.VOCAB_SIZE) )
            tok_scores_one.append( 1/(sum(self.counts_one.values()) + self.VOCAB_SIZE) )
        # if in both classes, just apply smoothing to both
        else:
            tok_scores_zero.append( (self.counts_zero[tok] + 1) /(sum(self.counts_zero.values()) + self.VOCAB_SIZE) )
            tok_scores_one.append( (self.counts_one[tok] + 1)/(sum(self.counts_one.values()) + self.VOCAB_SIZE) )

    # calculate priors: ( prior(class0) , prior(class1) )
    prior  = ( self.n_zeros/self.n_docs , 1- (self.n_zeros/self.n_docs) )

    # calculate the sum of logs for scores of all tokens
    return { "0" : (np.e ** (np.log(prior[0]) + np.sum(np.log(tok_scores_zero)))) ,  "1" : (np.e ** (np.log(prior[1]) + np.sum(np.log(tok_scores_one))) ) }


  def classify(self, data):
    """
    Label a given piece of text
    Parameters:
      data - str like "I loved the hotel"
    Return: string class label
    """
    scores = self.score(data)
    if scores["0"] == scores["1"]:
        return "0"
    max_class = max(scores, key=scores.get)
    return max_class


  def featurize(self, data):
    """
    we use this format to make implementation of part 1.3 more straightforward and to be
    consistent with what you see in nltk
    Parameters:
      data - str like "I loved the hotel"
    Return: a list of tuples linking features to values
    for BoW, a list of tuples linking every word to True [("I", True), ("loved", True), ("it", True)]
    """
    # we are not implementing featurizer for the baseline class
    # however, within the train function we implement BoW
    return [()]

  def __str__(self):
    return "Naive Bayes - bag-of-words baseline"


class TextClassifyImproved:

  def __init__(self):

    self.featurizer = "bow"


  def sigmoid(self, z):
    return 1 / (1 + np.exp(-z))

  def loss(self, y_hat, y):
    loss = -np.mean(y*(np.log(y_hat)) - (1-y)*np.log(1-y_hat))
    return loss

  def train(self, examples):
    """
    Trains the classifier based on the given examples
    Parameters:
      examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    Return: None
    """
    # individual_sentences = [ex[1] for ex in examples]
    # individual_words = [individual_sentence.split() for individual_sentence in individual_sentences]
    # self.vocabulary = set(chain.from_iterable(individual_words))
    sentences = [ex[1].split() for ex in examples]
    self.vocabulary = set(chain.from_iterable(sentences))
    self.VOCAB_SIZE = len(self.vocabulary)
    # print("LOGISTIC REGRESSION :", individual_sentences[0])
    #Now we have sentences > figure out how to train the model > Gradient descent function

    # run featurization once to obtain feature_size
    _ = self.featurize(examples[0][1])

    self.w = np.zeros(self.feature_size)
    self.b = 0.1
    self.learning_rate = 0.1
    self.iterations = 10 #Change no of iterations if needed

    for i in range(self.iterations):
        for ex in examples:
            self.x = self.featurize(ex[1])
            self.y = int(ex[2])
            z = np.dot(self.x, self.w) + self.b
            y_hat = self.sigmoid(z)
            #J = self.loss(y_hat, y) #causing errors  divide by zero encountered in log & invalid value encountered in double_scalars
            dw = np.dot(self.x.T, (y_hat - self.y)) #/ y.shape[0]  represents the no of examples
            db = np.sum((y_hat - self.y)) #/ y.shape[0]
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    return None

  def score(self, data):
    """
    Score a given piece of text
    youâ€™ll compute e ^ (log(p(c)) + sum(log(p(w_i | c))) here

    Parameters:
      data - str like "I loved the hotel"
    Return: dict of class: score mappings
    return a dictionary of the values of P(data | c)  for each class,
    as in section 4.3 of the textbook e.g. {"0": 0.000061, "1": 0.000032}
    """

    self.x = self.featurize(data)
    z = np.dot(self.x, self.w) + self.b
    probability = self.sigmoid(z)

    return { "0" : 1-probability ,  "1" : probability }


  def classify(self, data):
    """
    Label a given piece of text (this is argmax of the score)
    Parameters:
      data - str like "I loved the hotel"
    Return: string class label
    """
    scores = self.score(data)
    if scores["0"] == scores["1"]:
        return "0"
    max_class = max(scores, key=scores.get)
    return max_class


  def featurize(self, data):
    """
    we use this format to make implementation of part 3 more straightforward and to be
    consistent with what you see in nltk
    Parameters:
      data - str like "I loved the hotel"
    Return: a list of tuples linking features to values
    for BoW, a list of tuples linking every word to True [("I", True), ("loved", True), ("it", True)]
    """
    tokens = data.split()
    list_of_positives = ["good","best","nice","great","awesome","love","gorgeous","fantastic","amazing","wonderful","incredible","spacious","delicious","accomodating","kudos","clean","safe"]
    list_of_negatives = ["horrible","terrible","worst","bad","worse","dirty","dislike","hate","hated","unsafe","disgusting","poor"]
    # Use zip function?

    bow = []
    frequency = Counter(tokens)
    feat = self.featurizer

    # only BoW
    if feat == "bow":
        for vocab_word in self.vocabulary:
            bow.append(frequency[vocab_word])

    # appends pos_count to the BoW features
    if feat == "bow-pos":
        for vocab_word in self.vocabulary:
            bow.append(frequency[vocab_word])
        counter = 0
        for pos in list_of_positives:
            if pos in list(frequency.keys()):
                counter += frequency[pos]
        bow.append(counter)

    # appends neg_count to the BoW features
    if feat == "bow-neg":
        for vocab_word in self.vocabulary:
            bow.append(frequency[vocab_word])
        counter = 0
        for neg in list_of_negatives:
            if neg in list(frequency.keys()):
                counter += frequency[neg]
        bow.append(counter)

    # appends num_all_capital_words to the BoW features
    if feat == "bow-capital":
        for vocab_word in self.vocabulary:
            bow.append(frequency[vocab_word])
        counter = 0
        for tok in list(frequency.keys()):
            if tok.isupper():
                counter += frequency[tok]
        bow.append(counter)


    if feat == "bow-length":
        for vocab_word in self.vocabulary:
            bow.append(frequency[vocab_word])
        bow.append(len(tokens))

    # appends both pos_count and neg_count to the BoW features
    if feat == "bow-pos-neg":
        for vocab_word in self.vocabulary:
            bow.append(frequency[vocab_word])
        counter = 0
        for pos in list_of_positives:
            if pos in list(frequency.keys()):
                counter += frequency[pos]
        bow.append(counter)
        counter = 0
        for neg in list_of_negatives:
            if neg in list(frequency.keys()):
                counter += frequency[neg]
        bow.append(counter)

    if feat == "pos-neg":
        counter = 0
        for pos in list_of_positives:
            if pos in list(frequency.keys()):
                counter += frequency[pos]
        bow.append(counter)
        counter = 0
        for neg in list_of_negatives:
            if neg in list(frequency.keys()):
                counter += frequency[neg]
        bow.append(counter)

    if feat == "pos-neg-capital":
        counter = 0
        for pos in list_of_positives:
            if pos in list(frequency.keys()):
                counter += frequency[pos]
        bow.append(counter)
        counter = 0
        for neg in list_of_negatives:
            if neg in list(frequency.keys()):
                counter += frequency[neg]
        bow.append(counter)
        counter = 0
        for tok in list(frequency.keys()):
            if tok.isupper():
                counter += frequency[tok]
        bow.append(counter)

    if feat == "bow-pos-neg-capital":
        for vocab_word in self.vocabulary:
            bow.append(frequency[vocab_word])
        counter = 0
        for pos in list_of_positives:
            if pos in list(frequency.keys()):
                counter += frequency[pos]
        bow.append(counter)
        counter = 0
        for neg in list_of_negatives:
            if neg in list(frequency.keys()):
                counter += frequency[neg]
        bow.append(counter)
        counter = 0
        for tok in list(frequency.keys()):
            if tok.isupper():
                counter += frequency[tok]
        bow.append(counter)

    if feat == "bow-pos-neg-capital-length":
        for vocab_word in self.vocabulary:
            bow.append(frequency[vocab_word])
        counter = 0
        for pos in list_of_positives:
            if pos in list(frequency.keys()):
                counter += frequency[pos]
        bow.append(counter)
        counter = 0
        for neg in list_of_negatives:
            if neg in list(frequency.keys()):
                counter += frequency[neg]
        bow.append(counter)
        counter = 0
        for tok in list(frequency.keys()):
            if tok.isupper():
                counter += frequency[tok]
        bow.append(counter)
        bow.append(len(tokens))

    # store the feature dimension to determine training parameters
    self.feature_size = len(bow)

    return np.array(bow)

  def __str__(self):
    return "Logistic Regression Classifier"


def k_fold(all_examples, k):
    # all_examples is a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    # containing all examples from the train and dev sets
    # return a list of lists containing k sublists where each sublist is one "fold" in the given data
    all_examples_split = list()
    all_examples_copy = list(all_examples)
    fold_size = int(len(all_examples) / k)
    for i in range(k):
        #reinitialize for every iteration
        fold = list()
        while len(fold)<fold_size:
            index = randrange(len(all_examples_copy))
            fold.append(all_examples_copy.pop(index))

        all_examples_split.append(fold)
    # print("K FOLD SPLIT :", all_examples_split)

    return all_examples_split


def main():
  training = sys.argv[1]
  testing = sys.argv[2]

  classifier = TextClassify()
  print(" -----  MODEL USED: -----")
  print(classifier)

  # do the things that you need to with your base class
  train_examples = generate_tuples_from_file(training)
  test_examples = generate_tuples_from_file(testing)

  classifier.train(train_examples)

  predictions, gold_labels = [], []

  for tst in test_examples:
    score = classifier.score(tst[1])
    pred = classifier.classify(tst[1])
    gold = tst[2]
    predictions.append(pred)
    gold_labels.append(gold)
    #print(f"\t\t\t[ INFO ] score for sentence: {tst}:{score} - prediction:{pred} - gold_lable:{gold}")

  # report precision, recall, f1
  print("\t\t[ INFO ] precision = ", precision( gold_labels , predictions))
  print("\t\t[ INFO ] recall = ", recall( gold_labels , predictions))
  print("\t\t[ INFO ] f1 = ", f1( gold_labels , predictions))

  # initial featurizer is BoW - we'll change this within a for loop
  improved = TextClassifyImproved()
  print(" -----  MODEL USED: -----")
  print(improved)
  # do the things that you need to with your improved class

  # try different featurizers:
  #for feat in ["bow","bow-pos","bow-neg","bow-capital","bow-length","bow-pos-neg","pos-neg","pos-neg-capital","bow-pos-neg-capital","bow-pos-neg-capital-length"]:
  for feat in ["bow"]:
      improved.featurizer = feat
      print(f"\t[ INFO ] featurizer: {feat}")
      improved.train(train_examples)

      predictions_lr, gold_labels_lr = [], []

      for tst in test_examples:
        score = improved.score(tst[1])
        pred = improved.classify(tst[1])
        gold = tst[2]
        predictions_lr.append(pred)
        gold_labels_lr.append(gold)
        #print(f"\t\t\t[ INFO ] score for sentence: {tst}:{score} - prediction:{pred} - gold_lable:{gold}")

      # report precision, recall, f1
      print("\t\t[ INFO ] precision = ", precision( gold_labels_lr , predictions_lr))
      print("\t\t[ INFO ] recall = ", recall( gold_labels_lr , predictions_lr))
      print("\t\t[ INFO ] f1 = ", f1( gold_labels_lr , predictions_lr))


  #kfold
  # combining two files
  train_examples.extend(test_examples)
  k_fold_data=k_fold(train_examples,10)
  print(" -----  K-Fold for Naive Bayes: -----")
  classifier = TextClassify()
  print(classifier)
  for fold in k_fold_data:
      trainSet = list(k_fold_data)
      trainSet.remove(fold)
      trainSet = sum(trainSet, [])
      testSet = list()
      for row in fold:
          rowCopy = tuple(row)
          testSet.append(rowCopy)
      classifier.train(trainSet)
      predictions_k_fold, gold_labels_k_fold = [], []
      for tst in testSet:
          score = classifier.score(tst[1])
          pred = classifier.classify(tst[1])
          gold = tst[2]
          predictions_k_fold.append(pred)
          gold_labels_k_fold.append(gold)
      print("PREDICTION LABELS :", predictions_k_fold)
      print("GOLD LABELS       :", gold_labels_k_fold)

      # print(f"[ INFO ] score for sentence: {tst}:{score} - prediction:{pred} - gold_lable:{gold}")
      # report precision, recall, f1
      print("[ INFO ] precision = ", precision( gold_labels_k_fold , predictions_k_fold))
      print("[ INFO ] recall = ", recall( gold_labels_k_fold , predictions_k_fold))
      print("[ INFO ] f1 = ", f1( gold_labels_k_fold , predictions_k_fold))



  print(" -----  K-Fold for Logistic Regression: -----")
  improved = TextClassifyImproved()
  print(improved)
  for fold in k_fold_data:
      trainSet = list(k_fold_data)
      trainSet.remove(fold)
      trainSet = sum(trainSet, [])
      testSet = list()
      for row in fold:
          rowCopy = tuple(row)
          testSet.append(rowCopy)
      improved.train(trainSet)
      predictions_k_fold, gold_labels_k_fold = [], []
      for tst in testSet:
          score = improved.score(tst[1])
          pred = improved.classify(tst[1])
          gold = tst[2]
          predictions_k_fold.append(pred)
          gold_labels_k_fold.append(gold)
      print("PREDICTION LABELS :", predictions_k_fold)
      print("GOLD LABELS       :", gold_labels_k_fold)

      # print(f"[ INFO ] score for sentence: {tst}:{score} - prediction:{pred} - gold_lable:{gold}")
      # report precision, recall, f1
      print("[ INFO ] precision = ", precision( gold_labels_k_fold , predictions_k_fold))
      print("[ INFO ] recall = ", recall( gold_labels_k_fold , predictions_k_fold))
      print("[ INFO ] f1 = ", f1( gold_labels_k_fold , predictions_k_fold))


if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage:", "python textclassify_model.py training-file.txt testing-file.txt")
    sys.exit(1)

  main()
