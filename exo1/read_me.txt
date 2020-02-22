README FILE
--- NLP EXERCISE 1
--- Johnny Chen, Guillaume Biagi

The function updating the words in our code is update_word: it takes as inputs the current word we are reading, the context words around it (in a given window) and some negative words and updates them by applying a gradient descent of the sigmoid function with a given learning rate. This function is then called on each word of each sentence in the train set.
The sampling to get the negative words is explained in the pdf file.
The parameters of our model are saved in json files except for the matrices which are saved in .npy files.
For the similarity, we just take the scalar product between the two words normalized by their norms (similar to a cosine). If one of the words is not in the models vocabulary, we return 0.
Our code also includes the test functions we used to test some functions.
