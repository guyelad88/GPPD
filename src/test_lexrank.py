from time import gmtime, strftime
import pandas as pd
import logging
import json

from calculate_word_contribute import CalculateWordContribute
from config import lexrank_config
from summarizer import LexRank

SUMMARY_SIZE = lexrank_config.test_lexrank['summary_size']
HTML_SUMMARY_SIZE = lexrank_config.test_lexrank['HTML_summary_size']

THRESHOLD = lexrank_config.test_lexrank['threshold']
DAMPING_FACTOR = lexrank_config.test_lexrank['damping_factor']
SUMMARIZATION_SIMILARITY_THRESHOLD = lexrank_config.test_lexrank['summarization_similarity_threshold']
CORPUS_SIZE = lexrank_config.test_lexrank['corpus_size']

PERSONALITY_WORD_FLAG = lexrank_config.test_lexrank['personality_word_flag']
RANDOM_WALK_FLAG = lexrank_config.test_lexrank['random_walk_flag']

PRODUCTS_IDS = lexrank_config.test_lexrank['products_ids']      # id's to run algorithm on

LEX_RANK_ALGORITHM_VERSION = lexrank_config.test_lexrank['lex_rank_algorithm_version']
PERSONALITY_TRAIT_DICT = lexrank_config.test_lexrank['personality_trait_dict']

# data paths
CORPUS_PATH_FILE = lexrank_config.test_lexrank['corpus_path_file']
TARGET_ITEM_DESCRIPTION_FILE = lexrank_config.test_lexrank['target_item_description_file']
TRAIT_RELATIVE_PATH_DICT = lexrank_config.test_lexrank['trait_relative_path_dict']

logging.getLogger().setLevel(logging.INFO)

log_dir = 'log/'
html_dir = '../results/lexrank/html/'


class WrapperLexRank:

    """
    Main function of the algorithm - run personalized-LexRank algorithm

    :argument
    summary_size: number of senetnces in the final summarization. 'max' or int
    threshold: min similarity between two node (sentences) in the graph. An edge is discarded if sim is smaller than threshold.
    damping_factor: probability not to jump (low -> 'personalized' jump often occur)
    summarization_similarity_threshold = max similarity of two sentences in the final description.

    personality_trait_dict: user personality ('H'/'L' assign to each trait)

    corpus_size = 'max'
    personality_word_flag:          if combine
    random_walk_flag:               if combine "personalized" jump (matrix).
    lex_rank_algorithm_version:     'personality-based-LexRank', 'vanilla-LexRank'

    corpus_path_file: file contain description to calculate idf from
    target_item_description_file: clean description (Amazon product) to run LexRank on them

    log_dir = 'log/'
    html_dir = 'html/'

    :raises

    :returns

    """
    def __init__(self):

        self.cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        self.log_file_name = None
        self.word_cont_dict = dict()    # word and correspond contribute value

    def check_input(self):
        for c_trait, user_value in PERSONALITY_TRAIT_DICT.items():
            if c_trait not in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
                raise ValueError('unknown trait: {}'.format(c_trait))
            if user_value not in ['H', 'L']:
                raise ValueError('unknown value for {}: {}'.format(c_trait, user_value))

        if DAMPING_FACTOR < 0 or DAMPING_FACTOR > 1:
            raise ValueError('damping factor must be a number between 0 to 1')

        if SUMMARIZATION_SIMILARITY_THRESHOLD < 0 or SUMMARIZATION_SIMILARITY_THRESHOLD > 1:
            raise ValueError('summarization similarity threshold must be a float between 0 to 1')

        if LEX_RANK_ALGORITHM_VERSION not in ['vanilla-LexRank', 'personality-based-LexRank']:
            raise ValueError('unknown lexrank_algorithm_version')

        if CORPUS_SIZE != 'max' and not isinstance(CORPUS_SIZE, int):
            raise ValueError('unknown corpus size variable')

        if CORPUS_SIZE != 'max' and CORPUS_SIZE < 20:
            raise ValueError('too small corpus size')

    def test_lexrank(self):

        self.check_input()    # check inputs are valid

        # AC_p: calculate words contribution for a given personality
        word_contibute_obj = CalculateWordContribute(TRAIT_RELATIVE_PATH_DICT,
                                                     PERSONALITY_TRAIT_DICT,
                                                     self.cur_time)
        word_contibute_obj.calculate_user_total_word_contribute()
        word_cont_dict = word_contibute_obj.meta_word_contribute        # Word contribution to user personality

        # corpus data (e.g. products descriptions - for the calculation of IDF later)
        documents = pd.read_csv(CORPUS_PATH_FILE, usecols=['description'], dtype={'description': str})['description'].dropna().tolist()

        # load item description to summarize -> csv with cols: ID (product_id), TITLE (product title), DESCRIPTION (product description)
        target_description_df = pd.read_csv(
            TARGET_ITEM_DESCRIPTION_FILE,
            usecols=['ID', 'TITLE', 'DESCRIPTION'],
            dtype={'ID': str}
        )

        for index, row in target_description_df.iterrows():

            desc_id = row['ID']
            desc_title = row['TITLE']                            # .encode('ascii', 'ignore')
            target_sentences = json.loads(row['DESCRIPTION'])
            target_sentences = [s for s in target_sentences]

            summary_size = len(target_sentences) if SUMMARY_SIZE == 'max' else min(SUMMARY_SIZE, len(target_sentences))

            logging.info('')
            logging.info('Description ID: {}, Length: {}, Title: {}'.format(
                desc_id, len(target_sentences), desc_title)
            )
            logging.info('item sentences: {}'.format(str(len(target_sentences))))

            # Personalized LexRank class obj
            lxr = LexRank(
                documents,
                LEX_RANK_ALGORITHM_VERSION,
                html_dir=html_dir,
                user_personality_dict=PERSONALITY_TRAIT_DICT,
                word_cont_dict=word_cont_dict,
                personality_word_flag=PERSONALITY_WORD_FLAG,        # similarity based personality
                random_walk_flag=RANDOM_WALK_FLAG,
                damping_factor=DAMPING_FACTOR,
                summarization_similarity_threshold=SUMMARIZATION_SIMILARITY_THRESHOLD,
                cur_time=self.cur_time,
                desc_title=desc_title,
                desc_id=desc_id,
                corpus_size=CORPUS_SIZE,
                stopwords=None,
                keep_numbers=False,
                keep_emails=False,
                include_new_words=True,
            )

            summary, sorted_ix, lex_scores, description_summary_list, discarded_sentences_list = lxr.get_summary(
                target_sentences,
                summary_size,
                THRESHOLD)

            # write results into log file
            self.log_results(summary, sorted_ix, lex_scores, description_summary_list, discarded_sentences_list)

            # build a sentences
            lxr.build_summarization_html()

    # log results after finish algorithm
    def log_results(self, summary, sorted_ix, lex_scores, description_summary_list, discarded_sentences_list):
        '''
        log results of LexRank algorithm
        '''
        summary_below_threshold = list()
        sorted_ix_below_threshold = list()

        logging.info('')
        logging.info('summary extracted:')
        for sen_idx, sentence in enumerate(summary):
            logging.info('idx: {}, score: {} - {}'.format(
                str(sorted_ix[sen_idx]), str(round(lex_scores[sorted_ix[sen_idx]], 3)), str(sentence.encode('utf-8'))
            ))
            if sorted_ix[sen_idx] not in discarded_sentences_list:
                summary_below_threshold.append(sentence.encode('utf-8'))
                sorted_ix_below_threshold.append(sorted_ix[sen_idx])

        logging.info('')
        logging.info('sentence order:')
        logging.info(sorted_ix)
        logging.info('')
        logging.info('sentence rank:')
        logging.info(lex_scores)
        logging.info('')
        logging.info('Summary output')
        logging.info(description_summary_list)

        # self.log_html_format(summary, sorted_ix, discarded_sentences_list)
        self.log_html_format(summary_below_threshold, sorted_ix_below_threshold)
        # self.log_html_format(summary_below_threshold, sorted_ix_below_threshold)

    @staticmethod
    def log_html_format(summary, sorted_ix):
        """
        :return: summary in HTML format, in length K (HTML_SUMMARY_SIZE) and ordered properly
        """

        relevant_summary = summary[:HTML_SUMMARY_SIZE]  # remain only first K sentences
        relevant_sorted_ix = sorted_ix[:HTML_SUMMARY_SIZE]  # remain only first K sentences original place

        def sort_list(list1, list2):
            zipped_pairs = zip(list2, list1)
            z = [x for _, x in sorted(zipped_pairs)]
            return z

        ordered_summary = sort_list(relevant_summary, relevant_sorted_ix)
        ordered_summary = [sen.decode("utf-8") for sen in ordered_summary]
        logging.info('')
        logging.info(LEX_RANK_ALGORITHM_VERSION)
        logging.info(PERSONALITY_TRAIT_DICT)
        logging.info('')
        logging.info('Summary in HTML format')
        logging.info(". <br/>".join(ordered_summary) + ".")
        logging.info('')


def main():
    LexRankObj = WrapperLexRank()
    LexRankObj.test_lexrank()


if __name__ == '__main__':
    main()
