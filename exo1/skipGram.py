from __future__ import division
import argparse
import pandas as pd

# useful stuff
import numpy as np

from scipy.special import expit
from sklearn.preprocessing import normalize

# our imports
import json

__authors__ = ["Johnny Chen", "Guillaume Biagi"]
__emails__ = []


def text2sentences(path):
    # feel free to make a better tokenization/pre-processing
    sentences = []
    with open(path) as f:
        for l in f:
            sentences.append(l.lower().split())
    return sentences


def loadPairs(path):
    data = pd.read_csv(path, delimiter="\t")
    pairs = zip(data["word1"], data["word2"], data["similarity"])
    return pairs


# w will be the vector of the center word
# wc will be the vector of the context word
# z will be the vector of the negative word


def update_words(w, wc, array_zj, lr=0.1):
    uc = np.dot(w, wc)
    array_nj = [np.dot(w, z) for z in array_zj]

    array_grad_loss_nj = expit(array_nj)  # scalar
    grad_loss_uc = -expit((-1) * uc)  # scalar
    array_grad_loss_zj = np.array(
        [grad_nj * w for grad_nj in array_grad_loss_nj]
    )  # array of vectors
    grad_loss_wc = grad_loss_uc * w  # vector
    grad_loss_w = grad_loss_uc * wc + sum(
        [array_grad_loss_nj[j] * array_zj[j, :] for j in range(len(array_zj))]
    )  # vector

    w = w - lr * grad_loss_w
    wc = wc - lr * grad_loss_wc
    array_zj = array_zj - lr * array_grad_loss_zj

    return (w, wc, array_zj)


def test_update_words():
    lr = 0.1
    w = np.array([1, 0, 0, 0])
    wc = np.array([1, 1, 0, 0])
    zj = np.array([0, 1, 1, 0])
    array_z = np.array([zj])

    uc = np.dot(w, wc)
    nj = np.dot(w, zj)

    wc_expected = wc - (lr * (-expit((-1) * uc)) * w)
    array_z_expected = np.array([zj - lr * expit(nj) * w])
    w_expected = w - lr * (-expit((-1) * uc) * wc + expit(nj) * zj)

    w1, wc, array_z = update_words(w, wc, array_z)

    assert np.array_equal(wc_expected, wc)
    assert np.array_equal(array_z_expected, array_z)
    assert np.array_equal(w_expected, w1)


def loss(w, wc, array_zj):
    uc = np.dot(w, wc)
    array_nj = [np.dot(w, z) for z in array_zj]

    p1 = expit(uc)
    p2 = np.product(-expit(array_nj))

    loss = -np.log(p1 * p2)

    return loss


class SkipGram:
    def __init__(
        self, sentences, nEmbed=100, negativeRate=5, window_size=5, minCount=5
    ):
        # sentences is an array of arrays of words
        self.word2id = {}  # word to ID mapping
        self.id2word = {}
        self.word2occurences = {}
        self.word2negative_sampling_probabilities = {}
        self.trainset = []  # set of sentences
        self.trainWords = 0
        self.accLoss = 0
        self.loss = []
        self.vocab = []  # list of valid words
        self.total_number_of_words = 0
        self.negative_rate = negativeRate
        self.window_size = window_size

        for array_of_words in sentences:
            for word in array_of_words:
                if self.word2id.get(word) is None:
                    self.word2id[word] = self.total_number_of_words
                    self.id2word[self.total_number_of_words] = word
                    self.word2occurences[word] = 1
                    self.total_number_of_words += 1
                else:
                    self.word2occurences[word] += 1

        self.vocab = list(self.word2id.keys())

        negative_sample_proba = self.create_negative_sample_probabilities(
            list(self.word2occurences.values())
        )
        self.word2negative_sampling_probabilities = dict(
            zip(self.word2occurences.keys(), negative_sample_proba)
        )

        train_ratio = 0.8
        self.trainset = sentences[0 : int(train_ratio * len(sentences))]

        # center_matrix will be the matrix containing the embeddings of the center words.
        self.center_matrix = np.zeros((self.total_number_of_words, nEmbed))

        # context_matrix will be the matrix containing the embeddings of the context words.
        self.context_matrix = np.zeros((self.total_number_of_words, nEmbed))

    def create_negative_sample_probabilities(self, occurences):
        occurences_with_power = np.power(occurences, 3 / 4)
        s = np.sum(occurences_with_power)
        probabilities = occurences_with_power / s
        return probabilities

    def sample(self, omit_ids, n_sampling=5):
       	random_values = np.random.rand(n_sampling)
        random_values.sort()
	
        negative_ids = []
	
        words_not_omitted = self.word2occurences.copy()
        for omit_id in omit_ids:
            omit_word = self.id2word[omit_id]
            del words_not_omitted[omit_word]
        id_not_omitted = [self.word2id[word] for word in words_not_omitted]
        
        probabilities = self.create_negative_sample_probabilities(
            list(words_not_omitted.values())
        )
	
        cursor_proba = 0
        upper_bound = probabilities[0]
        # for word, probability in self.word2negative_sampling_probabilities:
        while len(random_values) != 0:
            while random_values[0] > upper_bound:
                cursor_proba += 1
                upper_bound += probabilities[cursor_proba]
		
            random_values = random_values[1:]
            negative_ids.append(id_not_omitted[cursor_proba])

        return negative_ids

    def train(self):
        for counter, sentence in enumerate(self.trainset):
            sentence = [w for w in filter(lambda word: word in self.vocab, sentence)]

            for word_position, word in enumerate(sentence):
                word_id = self.word2id[word]
                window_size = np.random.randint(self.window_size) + 1
                start = max(0, word_position - window_size)
                end = min(word_position + window_size + 1, len(sentence))

                for context_word in sentence[start:end]:
                    ctxtId = self.word2id[context_word]
                    if ctxtId == word_id:
                        continue
                    negativeIds = self.sample({word_id, ctxtId})

                    self.trainWord(word_id, ctxtId, negativeIds)
                    self.trainWords += 1

            if counter % 1000 == 0:
                print(" > training %d of %d" % (counter, len(self.trainset)))
                self.loss.append(self.accLoss / self.trainWords)
                self.trainWords = 0
                self.accLoss = 0.0

    def trainWord(self, wordId, contextId, negativeIds):
        w = self.center_matrix[wordId, :]
        wc = self.context_matrix[contextId, :]
        array_zj = self.context_matrix[negativeIds, :]

        # we do not need to update other context words, as the gradient_loss_other_context_word = 0
        w, wc, array_zj = update_words(w, wc, array_zj, lr=0.1)

        self.center_matrix[wordId, :] = w
        self.context_matrix[contextId, :] = wc
        self.context_matrix[negativeIds, :] = array_zj

        # raise NotImplementedError("here is all the fun!")

    def save(self, path):
        # Pas utile de tout sauvegarder (on peut retrouver total_number_of_words ou sampling probabilities sans les sauvegarder).
        # A ameliorer plus tard si besoin.

        # Saving parameters. For now just putting everything in one file. Maybe discuss later if we separate
        with open(path + "params.json", "w") as json_file:
            json.dump(
                {
                    "word2id": self.word2id,
                    "word2occurences": self.word2occurences,
                    "word2negative_sampling_probabilities": self.word2negative_sampling_probabilities,
                    "vocab": self.vocab,
                    "trainset": self.trainset,
                    "total_number_of_words": self.total_number_of_words,
                    "negative_rate": self.negative_rate,
                    "window_size": self.window_size,
                },
                json_file,
            )

        np.save(path + "center_matrix.npy", self.center_matrix)
        np.save(path + "context_matrix.npy", self.context_matrix)

    def similarity(self, word1, word2):
        raise NotImplementedError("Not implemented yet")

    @staticmethod
    def load(path):
        sg = SkipGram(sentences=[])

        with open(path + "params.json", "r") as json_file:
            params = json.load(json_file)
            for key in params:
                setattr(sg, key, params.get(key))

        setattr(sg, "center_matrix", np.load(path + "center_matrix.npy"))
        setattr(sg, "context_matrix", np.load(path + "context_matrix.npy"))
        for (word,ids) in sg.word2id.items():
            sg.id2word[ids] = word

        return sg


def test_sample():
    text = "toto tata toto"
    sentences = [text.split()]  # just one sentence
    sg = SkipGram(sentences)
    tata_id = sg.word2id.get("tata")
    toto_id = sg.word2id.get("toto")

    samples = sg.sample([tata_id], n_sampling=100)
    assert tata_id not in samples

    samples = sg.sample([], n_sampling=100)
    assert (tata_id in samples) and (toto_id in samples)


if __name__ == "__main__":

    test_sample()
    test_update_words()

    parser = argparse.ArgumentParser()
    parser.add_argument("--text", help="path containing training data", required=True)
    parser.add_argument(
        "--model",
        help="path to store/read model (when training/testing)",
        required=True,
    )
    parser.add_argument("--test", help="enters test mode", action="store_true")

    opts = parser.parse_args()

    if not opts.test:

        text_path = (
            opts.text
            or "data/data/training-monolingual.tokenized.shuffled/news.en-00001-of-00100"
        )

        sentences = text2sentences(text_path)
        sg = SkipGram(sentences)
        sg.train()
        sg.save(opts.model or "params/")

    else:
        pairs = loadPairs(opts.text)
        sg = SkipGram.load(opts.model or "params/")
        for a, b, _ in pairs:
            # make sure this does not raise any exception, even if a or b are not in sg.vocab
            print(sg.similarity(a, b))
