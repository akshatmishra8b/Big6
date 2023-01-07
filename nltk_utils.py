import nltk
from nltk.stem import PorterStemmer
from typing import List

stemmer = PorterStemmer()

def tokenize(sentence: str) -> List[str]:
    """
    Tokenize a sentence into a list of words/tokens.
    A token can be a word or punctuation character, or number.
    """
    return nltk.word_tokenize(sentence)

def stem(word: str) -> str:
    """
    Find the root form of a word using stemmer.
    Examples:
        stem("organize") -> "organ"
        stem("organizes") -> "organ"
        stem("organizing") -> "organ"
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence: List[str], words: List[str]) -> List[int]:
    """
    Create a bag of words representation for a sentence.
    Return a list of 1s and 0s indicating the presence or absence of each word in the sentence.
    Example:
        tokenized_sentence = ["hello", "how", "are", "you"]
        words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
        bag_of_words(tokenized_sentence, words) -> [0, 1, 0, 1, 0, 0, 0]
    """
    # stem each word in the sentence
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = [0] * len(words)
    # set the value to 1 for each word that exists in the sentence
    for i, w in enumerate(words):
        if w in sentence_words:
            bag[i] = 1
    return bag
