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
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
import re

def get_raw_sentences(pdf_path):
    punkt_param = PunktParameters()
    punkt_param.abbrev_types = set(['dr', 'vs', 'mr', 'mrs', 'prof', 'inc', 'al', 'etc', 'e.g', 'i.e', 'fig'])
    sentence_splitter = PunktSentenceTokenizer(punkt_param)
    with open(pdf_path, "r") as file:
            # returns one string with all of the text
        raw_text = file.read()
    file.close()

    raw_sentences = sentence_splitter.tokenize(re.sub(" +", " ", raw_text))
    return raw_sentences

def get_clean_line(sentence):
    # removes any instance of an in-text citation following any of these formats:
    # (Smith & Johnson, 2019), (Smith, 2019), (Smith et al., 2019), (Smith & Johnson, 2019; James, 2019)
    clean_text = re.sub(r"\s\((?:(?:[\w \.&]+\, )+[0-9]{4}[;|:]*\s*)+\)", "", sentence)

    # need to remove instances of citations within sentences (e.g "Smith et al. (2018) said that....")
    # as these cause the sentences to get split up where they aren't supposed to
    # clean_text = re.sub("(et al\.)", "et al", clean_text)
    # clean_text = re.sub("(e\.g\.)", "e.g", clean_text)
    # clean_text = re.sub("(i\.e\.)", "i.e", clean_text)
    # clean_text = re.sub("(etc\.)", "etc", clean_text)
    # clean_text = re.sub("(Fig\.)", "Fig", clean_text)
    clean_text = re.sub("(Table \w+)", "Table", clean_text)


    # also need to remove numbers, includes decimals
    clean_text = re.sub(r"\d+(\.[0-9]+)*", "", clean_text)
    clean_text = re.sub(r"{.+}@\S+", "", clean_text)


    # removes any URLs
    #clean_text = re.sub(r"http\S+", "", clean_text)
    clean_text = re.sub(r'(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#“”"-]*[\w@?^=%&\/~+“”"#-])', " ", clean_text)
    clean_text = re.sub(r"www\.\S+", "", clean_text)

    # removes any additional white space (e.g: "I like      cats   .") 
    clean_text = re.sub(" +", " ", clean_text)
    return clean_text

def get_clean_sentences(raw_sentences):
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

    clean_sentences = []
    for sentence in raw_sentences:
        clean_sentences.append(get_clean_line(sentence))
    return clean_sentences

def get_tokenized_sentences(raw_sentences):
    # remove punctuation and make all letters lowercase
    clean_sentences = [re.sub(r'[^\w\s]','',sentence.lower()) for sentence in get_clean_sentences(raw_sentences)]

    stop_words = stopwords.words('english')

    # Removes stop words (using the list of stop words from NLTK) and returns
    # A list of lists, with each list containing the words in each sentence
    sentence_tokens = [[words for words in word_tokenize(sentence) if words not in 
                        stop_words] for sentence in clean_sentences]
    return sentence_tokens




