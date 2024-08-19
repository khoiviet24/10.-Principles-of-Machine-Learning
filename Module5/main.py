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

# Uncomment to download models 
"""
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

    # Peform random substitution using word embedding similarity
    augmented_texts = aug.augment(input)

    # Append result to output file
    with open('augmented_result.txt', 'a') as output_file:
        output_file.write('Random substitution using word embedding similarity:\n')
        for augmented_text in augmented_texts:
            output_file.write(augmented_text + '\n')
        output_file.write('\n')

def contextual_word_embedding(input):
    # Substitute word by contextual word embeddings (BERT, DistilBERT, RoBERTA or XLNet)
    aug = naw.ContextualWordEmbsAug(
    # Options: 'distilbert-base-uncased', 'roberta-base', etc.
    model_path = 'bert-base-uncased', 
    # Options: "substitute", or "insert"
    action = "substitute")

    # Perform subsitution by contextual word embedding
    augmented_texts = aug.augment(input)

    # Append result to output file
    with open('augmented_result.txt', 'a') as output_file:
        output_file.write('Substitution by contextual word embedding:\n')
        for augmented_text in augmented_texts:
            output_file.write(augmented_text + '\n')
        output_file.write('\n')

def wordnet_synonym(input):
    # Substitute word by WordNet's synonym.
    # Option: set the max number of words to replace with synonym.
    aug = naw.SynonymAug(aug_src = 'wordnet', aug_max = 3)

    # Perform WordNet's synonym substitution
    augmented_texts = aug.augment(input, )

    # Append result to output file
    with open('augmented_result.txt', 'a') as output_file:
        output_file.write('Substitution by WordNet\'s synonym:\n')
        for augmented_text in augmented_texts:
            output_file.write(augmented_text + '\n')
        output_file.write('\n')

def back_translation(input):
    # Use back translation augmenter
    back_translation_aug = naw.BackTranslationAug(
        from_model_name='facebook/wmt19-en-de', 
        to_model_name='facebook/wmt19-de-en'
    )

    # Perform back translation
    augmented_texts = back_translation_aug.augment(input)
    
    # Append result to output file
    with open('augmented_result.txt', 'a') as output_file:
        output_file.write('Back Translation:\n')
        for augmented_text in augmented_texts:
            output_file.write(augmented_text + '\n')
        output_file.write('\n')

def load_text_from_file(file_path):
    # Read text from a file
    with open(file_path, 'r') as file:
        text = file.read()

    return text

def main():
    # Load text
    file_path = 'text_dataset.txt'
    text = load_text_from_file(file_path)

    # Clear content of output file
    with open('augmented_result.txt', 'w') as file:
        file.write('')

    # Append Original text
    with open('augmented_result.txt', 'a') as output_file:
        output_file.write('Original text:\n')
        output_file.write(text + '\n\n')

    random_embedding_similarity(text)
    contextual_word_embedding(text)
    wordnet_synonym(text)
    back_translation(text)
    
# Execute main
if __name__ == "__main__":
    main()