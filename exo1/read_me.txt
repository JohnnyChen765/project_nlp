README FILE
--- NLP EXERCISE 1
--- Johnny Chen, Guillaume Biagi

The function updating the words in our code is update_word: it takes as inputs the current word we are reading, the context words around it (in a given window) and some negative words and updates them by applying a gradient descent of the sigmoid function with a given learning rate. This function is then called on each word of each sentence in the train set.
The sampling to get the negative words is explained in the pdf file.
The parameters of our model are saved in json files except for the matrices which are saved in .npy files.
For the similarity, we just take the scalar product between the two words normalized by their norms (similar to a cosine). If one of the words is not in the models vocabulary, we return 0.
Our code also includes the test functions we used to test some functions.

Reference:
https://towardsdatascience.com/an-implementation-guide-to-word2vec-using-numpy-and-google-sheets-13445eebd281
https://aegis4048.github.io/optimize_computational_efficiency_of_skip-gram_with_negative_sampling
https://towardsdatascience.com/skip-gram-nlp-context-words-prediction-algorithm-5bbf34f84e0c
http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/

Paper:
    word2vec Parameter Learning Explained
    Xin Rong
    ronxin@umich.edu

    Distributed Representations of Words and Phrases
    and their Compositionality