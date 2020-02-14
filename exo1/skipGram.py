from __future__ import division
import argparse
import pandas as pd

# useful stuff
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize


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

        negative_sample_proba = self.create_negative_sample_probabilities(
            list(self.word2occurences.values())
        )
        self.word2negative_sampling_probabilities = dict(
            zip(self.word2occurences.keys(), negative_sample_proba)
        )

        train_ratio = 0.8
        self.trainset = sentences[0 : int(train_ratio * len(sentences))]
        self.vocab = list(self.word2id.keys())

    def create_negative_sample_probabilities(self, occurences):
        occurences_with_power = np.power(occurences, 3 / 4)
        s = np.sum(occurences_with_power)
        probabilities = occurences_with_power / s
        return probabilities

    def sample(self, omit_ids, n_sampling=5):
        """samples negative words, ommitting those in set omit"""
        random_values = np.random.rand(n_sampling)
        random_values.sort()

        negative_ids = []

        words_not_omitted = self.word2occurences.copy()
        for omit_id in omit_ids:
            omit_word = self.id2word[omit_id]
            del words_not_omitted[omit_word]

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
            negative_ids.append(cursor_proba)

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
        raise NotImplementedError("here is all the fun!")

    def save(self, path):
        # Pas utile de tout sauvegarder (on peut retrouver total_number_of_words ou sampling probabilities sans les sauvegarder).
        # A ameliorer plus tard si besoin.
        # Saving ids
        file_id = open(path + "ids.txt", "w")
        for w in self.word2id:
            file_id.write(w + "," + str(self.word2id[w]) + "\n")

        # Saving occurences
        file_oc = open(path + "ocs.txt", "w")
        for w in self.word2occurences:
            file_oc.write(w + "," + str(self.word2occurences[w]) + "\n")

        # Saving sampling probabilities
        file_sp = open(path + "sps.txt", "w")
        for w in self.word2negative_sampling_probabilities:
            file_sp.write(
                w + "," + str(self.word2negative_sampling_probabilities[w]) + "\n"
            )

        # Saving trainset
        file_ts = open(path + "ts.txt", "w")
        for sentence in self.trainset:
            for word in sentence:
                file_ts.write(word + " ")
            file_ts.write("\n")

        # Saving vocabulary
        file_voc = open(path + "voc.txt", "w")
        for w in self.vocab:
            file_voc.write(w + ",")

        # Saving other parameters
        file_param = open(path + "param.txt", "w")
        file_param.write(
            str(self.total_number_of_words)
            + "\n"
            + str(self.negative_rate)
            + "\n"
            + str(self.window_size)
        )

    def similarity(self, word1, word2):
        """
			computes similiarity between the two words. unknown words are mapped to one common vector
		:param word1:
		:param word2:
		:return: a float \in [0,1] indicating the similarity (the higher the more similar)
		"""
        raise NotImplementedError("implement it!")

    @staticmethod
    def load(path):
        raise NotImplementedError("implement it!")


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

        test_sample()

        text_path = (
            opts.text
            or "data/data/training-monolingual.tokenized.shuffled/news.en-00001-of-00100"
        )

        sentences = text2sentences(text_path)
        sg = SkipGram(sentences)
        sg.train()
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)

        sg = SkipGram.load(opts.model)
        for a, b, _ in pairs:
            # make sure this does not raise any exception, even if a or b are not in sg.vocab
            print(sg.similarity(a, b))
