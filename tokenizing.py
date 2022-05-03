"""
This script contains functions for cleaning up pre-processed PDF text and
tokenizing its sentences.

Terminology Definitions:

raw_text:   The text ripped straight from the PDF. This text has been untouched,
            so it still contains all in-text citations, numbers, URLs, etc.

clean_text: This is the "cleaned up" version of the text. We remove in-text
            citations of most formats, numbers, URLs (http and www), and any
            additional white space.

sentences:  This is the clean_text but it's been split up into individual
            sentences, and are all stored as a list.

clean_sentences: The same as sentences, but all punctuation has been removed,
                 and the text is in all lower case.

sentence_tokens: These are the clean sentences that have had all of their stop
                 words removed. Stop words are words like: is, the, are, etc.
                 Each sentences is now a list of its important words.     
"""

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re


def get_clean_sentences(file_path):
    """
    Cleans pre-processed PDF text to remove any noise and converts the text
    into a list of sentences for further processing.

    Parameters
    ----------
    file_path : str
        The file path for the pre-processed PDF text

    Returns
    -------
    list
        A list of all of the sentences within the text
    """

    with open(file_path, "r") as file:
        # returns one string with all of the text
        raw_text = file.read()
    file.close()

    # this removes any instance of [<number>] as some papers use this as a
    # way of citing
    clean_text = re.sub(r"\[\d+\]", "", raw_text)

    # removes any instance of in-text citations following any of these formats:
    # (Smith & Johnson, 2019), (Smith, 2019), (Smith et al., 2019),
    # (Smith & Johnson, 2019; James, 2019)
    clean_text = re.sub(r"\((?:(?:[\w \.&]+\, )+[0-9]{4}[;|:]*\s*)+\)", "", clean_text)

    # need to remove instances of citations within sentences 
    # (e.g "Smith et al. (2018) said that....")
    # as these cause the sentences to get split up where they aren't supposed to
    clean_text = re.sub("(et al.)", "et al", clean_text)

    # removes numbers, includes decimals
    clean_text = re.sub(r"\d+\.*", " ", clean_text)

    # removes any URLs
    clean_text = re.sub(r"http\S+", "", clean_text)
    clean_text = re.sub(r"www\.\S+", "", clean_text)

    # removes any additional white space (e.g: "I like      cats   .") 
    clean_text = re.sub(" +", " ", clean_text)
    
    sentences = sent_tokenize(clean_text)
    return sentences

def get_tokenized_sentences(sentences):
    # remove punctuation and make all letters lowercase
    clean_sentences = [re.sub(r'[^\w\s]','',sentence.lower()) for sentence in sentences]

    stop_words = stopwords.words('english')

    # Removes stop words (using the list of stop words from NLTK) and returns
    # A list of lists, with each list containing the words in each sentence
    sentence_tokens = [[words for words in word_tokenize(sentence) if words not in 
                        stop_words] for sentence in clean_sentences]
    return sentence_tokens