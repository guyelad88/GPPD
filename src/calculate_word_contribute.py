import os
import pandas as pd
import statistics
import operator
import logging

from os import listdir
from os.path import isfile, join

logging.getLogger().setLevel(logging.INFO)

TOP_SHOW = 10


class CalculateWordContribute:
    """  """
    def __init__(self, trait_relative_path_dict, personality_trait_dict, time):

        self.trait_relative_path_dict = trait_relative_path_dict            # specific dir contain kl
        self.personality_trait_dict = personality_trait_dict                # 'H'/'L' to each personality
        self.cur_time = time

        self.meta_word_contribute = dict()
        self.meta_word_count = dict()               # number of word appearance
        self.meta_word_values_diff_trait = dict()   # word cont in different traits

        return

    # calculate word contribute to KL using merging all user personality trait value
    def calculate_user_total_word_contribute(self):
        logging.info('')
        for cur_trait, trait_value in self.personality_trait_dict.items():      # check high/low input
            logging.info('Personality trait: {}, Type: {}'.format(cur_trait, trait_value))
            cur_file_path = self.trait_relative_path_dict

            trait_file_suffix = [
                f for f in listdir(cur_file_path) if isfile(join(cur_file_path, f)) and cur_trait in f
            ]

            if trait_value == 'H':
                file_name = [s for s in trait_file_suffix if '{}_High.csv'.format(cur_trait) == s]
                assert len(file_name) == 1, "trait_High.csv is not exists, please add a file"
                cur_file_path = os.path.join(cur_file_path, file_name[0])

            elif trait_value == 'L':
                file_name = [s for s in trait_file_suffix if '{}_Low.csv'.format(cur_trait) == s]
                assert len(file_name) == 1, "trait_Low.csv is not exists, please add a file"
                cur_file_path = os.path.join(cur_file_path, file_name[0])

            else:
                raise ValueError('trait value must be H/L ({})'.format(cur_trait))

            # load excel file into df
            cur_trait_df = pd.read_csv(open(cur_file_path, 'rb'))
            assert {'Word', 'Word_contribution'}.issubset(cur_trait_df.columns), 'Word and Word_contribution columns must be exists'
            logging.info('num of words: {}'.format(cur_trait_df.shape[0]))

            for index, cur_row in cur_trait_df.iterrows():
                cur_word = cur_row['Word']
                cur_cont = cur_row['Word_contribution']

                # check word is a string
                if not isinstance(cur_word, str):
                    continue

                # check if current word first time seen (in respect to all OCEAN traits)
                if cur_word not in self.meta_word_contribute:
                    self.meta_word_contribute[cur_word] = 0.0   # word contribution in respect to user personality
                    self.meta_word_count[cur_word] = 0
                    self.meta_word_values_diff_trait[cur_word] = list()

                # update word total cont (moving average of old+new)
                prev_cont = self.meta_word_contribute[cur_word]
                prev_amount = self.meta_word_count[cur_word]
                new_cont = (prev_cont*prev_amount + cur_cont*1.0)/(prev_amount+1)
                self.meta_word_contribute[cur_word] = new_cont
                self.meta_word_count[cur_word] += 1
                self.meta_word_values_diff_trait[cur_word].append(round(cur_cont, 3))

        logging.info('normalize values after aggregate all trait values together')
        min_value = min(self.meta_word_contribute.values())
        max_value = max(self.meta_word_contribute.values())
        denominator = max_value - min_value
        for cur_word, cur_val in self.meta_word_contribute.items():
            self.meta_word_contribute[cur_word] = (cur_val-min_value)/denominator

        logging.info('')
        logging.info('words mean values: {}'.format(round(statistics.mean(self.meta_word_contribute.values()), 3)))

        self._log_word_contribute()

    # log top k most associated and unrelated words to user personality
    def _log_word_contribute(self):
        list_word_contribute_sort = sorted(self.meta_word_contribute.items(), key=operator.itemgetter(1))
        list_word_contribute_sort.reverse()

        logging.info('log top k={} associated and unrelated words to user personality'.format(TOP_SHOW))
        logging.info('Top associated words to user personality:')

        def print_word_inf(word_cont_tuple):
            return 'word: {}, contribution: {}, trait appear: {}, trait values: {}'.format(
                word_cont_tuple[0], round(word_cont_tuple[1], 3), self.meta_word_count[word_cont_tuple[0]], self.meta_word_values_diff_trait[word_cont_tuple[0]])

        for w_i, word_cont_tuple in enumerate(list_word_contribute_sort[:TOP_SHOW]):
            logging.info(print_word_inf(word_cont_tuple))

        list_word_contribute_sort.reverse()
        logging.info('Top unrelated words to user personality:')
        for w_i, word_cont_tuple in enumerate(list_word_contribute_sort[:TOP_SHOW]):
            logging.info(print_word_inf(word_cont_tuple))


if __name__ == '__main__':
    raise Exception('main is not support from here')
