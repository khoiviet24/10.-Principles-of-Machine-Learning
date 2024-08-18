import gensim
import transformers
import sacremoses
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
from nlpaug.util import Action

from transformers.utils import logging
logging.set_verbosity_error() 

"""
# Download models to a temporary path
from nlpaug.util.file.download import DownloadUtil
DownloadUtil.download_word2vec(dest_dir = '.')
# Possible values are ‘wiki-news-300d-1M’, ‘wiki-news-300d-1M-subword’, ‘crawl-300d-2M’ and ‘crawl-300d-2M-subword’
DownloadUtil.download_fasttext(dest_dir = '.', model_name = 'crawl-300d-2M')
# Possible values are ‘glove.6B’, ‘glove.42B.300d’, ‘glove.840B.300d’ and ‘glove.twitter.27B’
DownloadUtil.download_glove(dest_dir = '.', model_name = 'glove.6B')
"""

def random_embedding_similarity(input):
    # Substitute or insert word randomly using word embeddings similarity
    aug = naw.WordEmbsAug(
    # Options: "word2vec", "glove", or "fasttext" 
    model_type = 'fasttext', 
    model_path = 'crawl-300d-2M.vec',
    # Options: "substitute", or "insert"
    action = "substitute")

    # Augment the text
    print()
    print("Random substitution using word embedding similarity")
    augmented_text = aug.augment(input)
    print(augmented_text)
    print()

def contextual_word_embedding(input):
    # Substitute word by contextual word embeddings (BERT, DistilBERT, RoBERTA or XLNet)
    aug = naw.ContextualWordEmbsAug(
    # Options: 'distilbert-base-uncased', 'roberta-base', etc.
    model_path = 'bert-base-uncased', 
    # Options: "substitute", or "insert"
    action = "substitute")

    print()
    print("Substitution by contextual word embedding")
    augmented_text = aug.augment(input)
    print(augmented_text)
    print()

def wordnet_synonym(input):
    # Substitute word by WordNet's synonym.
    # Option: set the max number of words to replace with synonym.
    aug = naw.SynonymAug(aug_src = 'wordnet', aug_max = 3)

    print()
    print("Substitution by WordNet's synonym")
    augmented_text = aug.augment(input, )
    print("Augmented Text:")
    print(augmented_text)
    print()


def back_translation(input):
    # Use back translation augmenter
    back_translation_aug = naw.BackTranslationAug(
        from_model_name='facebook/wmt19-en-de', 
        to_model_name='facebook/wmt19-de-en'
    )
    print()
    print('Back Translation')
    print(back_translation_aug.augment(input))
    print()

def main():
    text = ' Is daily coffee consumption good for our health?  I guess it is reasonable to believe so, but it may also depend on how much you drink.'
    print('Original Text')
    print(text)
    print()

    random_embedding_similarity(text)
    contextual_word_embedding(text)
    wordnet_synonym(text)
    back_translation(text)
    
# Execute main
if __name__ == "__main__":
    main()