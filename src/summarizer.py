from colour import Color
import numpy as np
import logging
import math

from sklearn.feature_extraction.text import CountVectorizer

from power_method import stationary_distribution
from create_summary import Summarization

logging.getLogger().setLevel(logging.INFO)


# class LexRank build all algorithm objects - (Graph (markov matrix)) and use power method class to find the stationary distribution
class LexRank:
    """
    calculate LexRank (stationary distribution)
    1. build Graph - a markov matrix
    2. use power method to calculate stationary distribution

    :argument

    :returns

    :raises

    """
    def __init__(
        self,
        documents,                  # corpus entire data
        lex_rank_algorithm_version,
        html_dir=None,
        user_personality_dict=None,
        word_cont_dict=None,
        personality_word_flag=False,
        random_walk_flag=True,
        damping_factor=0.85,
        summarization_similarity_threshold=0.5,
        corpus_size='max',
        cur_time=None,
        desc_title=None,
        desc_id=None,
        stopwords=None,
        keep_numbers=False,
        keep_emails=False,
        keep_urls=False,
        include_new_words=True,
    ):
        if stopwords is None:
            self.stopwords = set()
        else:
            self.stopwords = stopwords

        self.keep_numbers = keep_numbers
        self.keep_emails = keep_emails
        self.keep_urls = keep_urls
        self.include_new_words = include_new_words
        self.vectorizer = None
        self.html_dir = html_dir
        self.cur_time = cur_time
        self.desc_title = desc_title        # item description title name
        self.desc_id = desc_id              # eBay item ID

        self.lex_rank_algorithm_version = lex_rank_algorithm_version

        self.personality_word_flag = personality_word_flag
        self.user_personality_dict = user_personality_dict

        self.word_cont_dict = word_cont_dict        # dictionary contain word and his contribute
        self.random_walk_flag = random_walk_flag    # random walk bool (insert into adjacency matrix)
        self.damping_factor = damping_factor        # probability to "random jump"
        # relevant for top-relevant summarization
        self.summarization_similarity_threshold = summarization_similarity_threshold
        self.corpus_size = corpus_size              # limitation of corpus size

        self.miss_idf = 0
        self.hit_idf = 0

        self.miss_word_contribute = 0
        self.hit_word_contribute = 0

        self.sentences = None       # sentences not sorted
        self.summary = None         # sentences sorted
        self.sorted_ix = None       # indexes sorted
        self.lex_scores = None      # lex score according to sentence index
        self.description_summary_list = None

        self.analyze = None
        self.color_list = None
        self.percentile_list = None
        self.list_avg_contribute = list()
        self.threshold = None               # edge min value threshold
        self.summary_size = None

        self.discarded_sentences = list()   # sentences which discarded from summary - similarity above threshold
        self.sentence_already_inserted = list()     # sentences in final summarization
        self.similarity_matrix_unnormalized = None  # similarity between each two sentences unnormalized

        self.doc_number_total = None
        self.unigram_corpus_features = None

        # calculate idf values on corpus data
        # idf_score - word and it's idf score
        # dict_word_index - word and it's index calculate in sklearn CountVectorizer
        self.idf_score, self.dict_word_index = self._calculate_idf(documents)

    # run variational LexRank algorithm
    def get_summary(
        self,
        sentences,          # item description input - sentences are not sorted of course
        summary_size=1,
        threshold=.03
    ):
        if not isinstance(summary_size, int) or summary_size < 1:
            raise ValueError('summary_size should be a positive integer: {}'.format(summary_size))

        logging.info('')
        logging.info('start get summary method')

        self.threshold = threshold
        self.summary_size = summary_size

        # run LexRank algorithm - inside split by variations
        lex_scores = self.rank_sentences(
            sentences,
            threshold=threshold
        )

        sorted_ix = np.argsort(lex_scores)[::-1]
        summary = [sentences[i] for i in sorted_ix[:summary_size]]

        # save for use next to create HTML
        self.sentences = sentences          # list of idx-sentence (not sorted)
        self.summary = summary              # list of sentence sorted by their lexrank score
        self.sorted_ix = sorted_ix          # list sentence idx sort by LexRank rate
        self.lex_scores = lex_scores        # list of idx-LexRank rate

        # summarize using LexRank results
        self.description_summary_list = self.run_summary_algorithm()    # call Summarization class

        return summary, sorted_ix, lex_scores, self.description_summary_list, self.discarded_sentences

    # body of algorithm
    def rank_sentences(
        self,
        sentences,
        threshold=.03
    ):
        # check valid threshold size
        if not isinstance(threshold, float) or not 0 <= threshold < 1:
            raise ValueError(
                '\'threshold\' should be a floating-point number '
                'from the interval [0, 1)',
            )

        logging.info('')
        logging.info('Rank sentences in LexRank')

        # original vectorizer (source library) - determine feature of splitting test sentences
        self.analyze = self.vectorizer.build_analyzer()

        # calculate tf score for test sentences
        tf_scores = list()                              # list of dict per test sentences
        for sentence in sentences:
            sentence_words = self.analyze(sentence)          # use original vectorizer to split test sentences data
            tf_scores.append(self._calculate_tf(sentence_words))        # insert tf score of test sentences

        # calculate adjacency matrix between sentences
        # adjacency matrix build from similarity matrix and jumping matrix
        similarity_matrix = self.calculate_adjacency_matrix(tf_scores)

        logging.info('build a markov matrix - normalize each row in similarity matrix to one')
        markov_matrix = self._markov_matrix(similarity_matrix)

        # compute stationary distribution
        logging.info('')
        logging.info('start to compute stationary distribution')
        scores = stationary_distribution(
            markov_matrix
        )
        logging.info('finish to compute stationary distribution using power method')

        return scores       # return stationary distribution

    # run Summarization class
    # create summary with respect to summarization approach - 'top_relevant', 'Bollegata', 'Shahaf'
    def run_summary_algorithm(self):

        # create summarization object
        summary_obj = Summarization(
            self.summarization_similarity_threshold,
            self.similarity_matrix_unnormalized,
            self.sorted_ix,
            self.summary_size,
            self.sentences
        )

        summary_obj.create_summary()
        self.discarded_sentences = summary_obj.discarded_sentences
        self.sentence_already_inserted = summary_obj.sentence_already_inserted

        return summary_obj.description_summary_list

    # adjacency_matrix is linear interpolation between similarity matrix and jump matrix
    def calculate_adjacency_matrix(self, tf_scores):

        logging.info('')
        logging.info('LexRank algorithm version: ' + str(self.lex_rank_algorithm_version))

        if self.lex_rank_algorithm_version == 'personality-based-LexRank':

            # similarity matrix
            logging.debug('')
            logging.debug('build personality-based similarity matrix:')
            self.similarity_matrix_unnormalized = self._calculate_similarity_matrix(tf_scores)
            similarity_matrix = self._markov_matrix(self.similarity_matrix_unnormalized)   # normalize matrix row to 1
            logging.debug('')
            logging.debug('similarity matrix (normalize to 1)')
            logging.debug(similarity_matrix)
            similarity_matrix = self.damping_factor * similarity_matrix     # SM: multiple with damping factor

            # jump matrix
            logging.info('')
            logging.info('build jump matrix: damping factor=' + str(self.damping_factor))
            jump_matrix = self._calculate_personality_based_jump_matrix(tf_scores)            # calculate jump matrix

            logging.debug('jump matrix (normalize to 1)')
            logging.debug(jump_matrix)

            jump_matrix = (1-self.damping_factor)*jump_matrix               # PM: multiple with 1-damping factor

            # adjacency matrix
            logging.info('')
            logging.info('build adjacency matrix - linear interpolation between similarity matrix and jump matrix')
            adjacency_matrix = similarity_matrix + jump_matrix

            logging.info('')
            logging.info('finish building adjacency matrix - ' + str(1-self.damping_factor) +
                              ' * PM + ' + str(self.damping_factor) + ' * SM')
            logging.debug(adjacency_matrix)

        elif self.lex_rank_algorithm_version == 'vanilla-LexRank':

            # similarity matrix
            logging.info('')
            logging.info('build Vanilla similarity matrix:')
            self.similarity_matrix_unnormalized = self._calculate_similarity_matrix(tf_scores)
            similarity_matrix = self._markov_matrix(self.similarity_matrix_unnormalized)  # normalize matrix row to 1
            logging.info('')
            logging.info('similarity matrix (normalize to 1)')
            logging.debug(similarity_matrix)
            similarity_matrix = (1 - self.damping_factor) * similarity_matrix  # multiple with damping factor

            # jump matrix
            logging.info('')
            logging.info('build jump matrix: damping factor=' + str(self.damping_factor))
            jump_matrix = self._calculate_jump_matrix(tf_scores)  # calculate jump matrix
            logging.debug('jump matrix (normalize to 1)')
            logging.debug(jump_matrix)
            jump_matrix = self.damping_factor * jump_matrix  # multiple with damping factor

            # adjacency matrix
            logging.info('')
            logging.info('build adjacency matrix - linear interpolation between similarity matrix and jump matrix')
            adjacency_matrix = similarity_matrix + jump_matrix

            logging.info('')
            logging.info('finish building adjacency matrix - ' + str(self.damping_factor) +
                              ' * U + ' + str(1 - self.damping_factor) + ' * SM')
            logging.debug(adjacency_matrix)

        return adjacency_matrix

    # calculate personality based jump matrix
    # more probability to sentences contain word correlated to user persoanlity
    def _calculate_personality_based_jump_matrix(self, tf_scores):
        logging.info('')
        logging.info('compute jump matrix (in probability 1-damping factor)')

        length = len(tf_scores)
        similarity_matrix = np.zeros([length] * 2)
        mean_word_cont = sum(self.word_cont_dict.values())/len(self.word_cont_dict.values())    # fill words didn't seen
        list_avg_contribute = list()

        for i in range(length):
            list_avg_contribute.append(self._calculate_average_word_contribute(tf_scores, i))

        # save to use in HTML function
        self.list_avg_contribute = list_avg_contribute

        list_normalize_avg_contribute = [x / sum(list_avg_contribute) for x in list_avg_contribute]
        x = np.array(list_normalize_avg_contribute)
        jump_matrix = np.ones((length, 1)) * x
        return jump_matrix

    # calculate jump personality with uniform probability
    def _calculate_jump_matrix(self, tf_scores):
        logging.info('')
        logging.info('compute jump matrix (in probability 1-damping factor)')

        length = len(tf_scores)
        ones_matrix = np.ones([length] * 2)
        uniform_matrix = ones_matrix / ones_matrix.sum(axis=1, keepdims=1)

        return uniform_matrix

    def _calculate_average_word_contribute(self, tf_scores, i):

        count_word_found = 0        # word with contribution
        count_word_contribute = 0
        sentence_dict = tf_scores[i]

        for cur_word, word_amount in sentence_dict.items():
            if cur_word in self.word_cont_dict:
                count_word_found += word_amount
                count_word_contribute += (self.word_cont_dict[cur_word]*word_amount)

        if count_word_found > 0:
            avg_contribute = float(count_word_contribute)/float(count_word_found)
        else:
            avg_contribute = sum(self.word_cont_dict.values())/len(self.word_cont_dict.values())

        logging.debug('i: ' + str(i) + ', avg contribute=' + str(round(avg_contribute, 4)))

        return avg_contribute

    ####################################
    # calculate meta IR word properties #
    # ####################################

    # calculate idf using corpus data & count vectorizer
    def _calculate_idf(self, documents):
        logging.info('calculate idf using corpus data with sklearn Count Vectorizer library')
        idf_score = dict()

        self.vectorizer = CountVectorizer(
            binary=True,
            preprocessor=None,
            tokenizer=None,
            stop_words='english',
            ngram_range=(1, 1),
            max_features=None,
            vocabulary=None,
            lowercase=True
        )

        if self.corpus_size != 'max' and self.corpus_size < len(documents):
            documents = documents[:self.corpus_size]

        unigram_vectorizer = self.vectorizer.fit_transform(documents)

        logging.info('Count vectorizer shape: Num descriptions {}, num words (features): {}'.format(unigram_vectorizer.shape[0], unigram_vectorizer.shape[1]))

        self.doc_number_total = unigram_vectorizer.shape[0]         # save for insertion to HTML file
        self.unigram_corpus_features = unigram_vectorizer.shape[1]

        dict_word_index = self.vectorizer.get_feature_names()

        for word_idx, word in enumerate(dict_word_index):
            doc_number_word = unigram_vectorizer[:, word_idx].sum()
            if doc_number_word > 0:
                idf_score[word] = math.log(float(self.doc_number_total) / float(doc_number_word))
            else:
                idf_score[word] = 0

        logging.info('finish calculating idf score')
        return idf_score, dict_word_index

    # calculate tf of words in test sentences (number of appearances)
    def _calculate_tf(self, tokenized_sentence):
        '''
        :param tokenized_sentence: words in the relevant sentence
        :return: dictionary with key: words in the sentence and value: word count.
        '''
        tf_score = {}

        for word in set(tokenized_sentence):
            tf = tokenized_sentence.count(word)
            tf_score[word] = tf

        return tf_score

    ####################################
    # calculate similarity matrix #
    # ####################################

    # calculate personality based similarity matrix between sentences - for Vanilla and Personality based approaches
    def _calculate_similarity_matrix(self, tf_scores):
        '''
        build a similarity matrix between sentences (next normalized to markov matrix)
        :param tf_scores: tf score for all sentences
        :return: similarity matrix
        '''
        length = len(tf_scores)

        similarity_matrix = np.zeros([length] * 2)

        logging.info('')
        logging.info('compute similarity matrix (in probability 1-damping factor)')
        logging.info('compute similarity between sentences:')

        for i in range(length):
            for j in range(i, length):

                similarity = self._idf_modified_cosine(tf_scores, i, j)

                # Symmetric matrix
                if similarity:
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity

        # self.plot_adjacency_matrix(similarity_matrix, tf_scores)

        # statistic about idf in test sentences does not appear in corpus
        logging.info('')
        logging.info('num miss words in idf list: ' + str(self.miss_idf))
        logging.info('num hit words in idf list: ' + str(self.hit_idf))

        if float(self.miss_idf)+float(self.hit_idf) > 0:
            logging.info('miss ratio: ' + str(round(float(self.miss_idf) /
                                                         (float(self.miss_idf)+float(self.hit_idf)), 3)))

        if self.lex_rank_algorithm_version == 'personality-based-LexRank':
            logging.info('')
            logging.info('num miss words in contribute dict: ' + str(self.miss_word_contribute))
            logging.info('num hit words in contribute dict: ' + str(self.hit_word_contribute))
            logging.info(
                'miss ratio: ' + str(round(float(self.miss_word_contribute) /
                                           (float(self.miss_word_contribute) + float(self.hit_word_contribute)), 3)))
            logging.info('count only words appear in both sentences for idf-similarity-cosine')
            logging.info('')
        logging.info('')
        logging.info('similarity matrix unnormalized:')
        logging.debug(similarity_matrix)
        return similarity_matrix

    # calculate weight between two sentences - for different LexRank algorithms
    def _idf_modified_cosine(self, tf_scores, i, j):
        '''
        compute similarity between two sentences - for Vanilla abnd Personality approaches
        :param tf_scores: tf_scores for all sentences (list of dict)
        :param i: sentence i index
        :param j: sentence j index
        :return: tf-idf-cosine-similarity between the two sentences
        '''
        if i == j:
            return 1

        tf_i, tf_j = tf_scores[i], tf_scores[j]
        words_i, words_j = set(tf_i.keys()), set(tf_j.keys())

        # compute nominator
        nominator = self.compute_nominator_idf_modified_cosine(words_i, words_j, tf_i, tf_j)
        '''
        nominator = 0

        for word in words_i & words_j:
            cur_idf_score = self.return_idf(word)
            nominator += tf_i[word] * tf_j[word] * cur_idf_score ** 2'''

        # compute denominator
        denominator_i, denominator_j = self.compute_denominator_idf_modified_cosine(words_i, words_j, tf_i, tf_j)
        '''
        denominator_i, denominator_j = 0, 0

        for word in words_i:
            cur_idf_score = self.return_idf(word)
            tfidf = tf_i[word] * cur_idf_score
            denominator_i += tfidf ** 2

        for word in words_j:
            cur_idf_score = self.return_idf(word)
            tfidf = tf_j[word] * cur_idf_score
            denominator_j += tfidf ** 2
        '''
        if math.sqrt(denominator_i) * math.sqrt(denominator_j) != 0:           # check denominator is not zero
            similarity = nominator / (math.sqrt(denominator_i) * math.sqrt(denominator_j))
        else:
            similarity = 0

        logging.debug('i,j: ' + str(i) + ',' + str(j) + ' similarity=' + str(round(similarity, 4)))
        return similarity

    # compute nominator of idf-modified-cosine
    def compute_nominator_idf_modified_cosine(self, words_i, words_j, tf_i, tf_j):
        nominator = 0
        for word in words_i & words_j:
            cur_idf_score = self.return_idf(word)

            # Personality-based-LexRank
            if self.lex_rank_algorithm_version == 'personality-based-LexRank':
                if word in self.word_cont_dict:
                    # similarity based personality
                    nominator += tf_i[word] * tf_j[word] * (self.word_cont_dict[word] ** 2) * (cur_idf_score ** 2)
                    self.hit_word_contribute += 1
                else:
                    nominator += tf_i[word] * tf_j[word] * cur_idf_score ** 2
                    self.miss_word_contribute += 1

            # Vanilla LexRank
            elif self.lex_rank_algorithm_version == 'vanilla-LexRank':
                nominator += tf_i[word] * tf_j[word] * cur_idf_score ** 2

        return nominator

    # compute denominator of idf-modified-cosine
    def compute_denominator_idf_modified_cosine(self, words_i, words_j, tf_i, tf_j):
        denominator_i, denominator_j = 0, 0
        for word in words_i:
            tfidf = None
            cur_idf_score = self.return_idf(word)
            if self.lex_rank_algorithm_version == 'personality-based-LexRank' and word in self.word_cont_dict:
                tfidf = tf_i[word] * cur_idf_score * self.word_cont_dict[word]
            elif self.lex_rank_algorithm_version == 'vanilla-LexRank':
                tfidf = tf_i[word] * cur_idf_score
            else:       # 'personality-based-LexRank' and word is not in KL distribution (item descriptions)
                tfidf = tf_i[word] * cur_idf_score
            denominator_i += tfidf ** 2

        for word in words_j:
            tfidf = None
            cur_idf_score = self.return_idf(word)
            if self.lex_rank_algorithm_version == 'personality-based-LexRank' and word in self.word_cont_dict:
                tfidf = tf_j[word] * cur_idf_score * self.word_cont_dict[word]
            elif self.lex_rank_algorithm_version == 'vanilla-LexRank':
                tfidf = tf_j[word] * cur_idf_score
            else:       # 'personality-based-LexRank' and word is not in KL distribution (item descriptions)
                tfidf = tf_j[word] * cur_idf_score
            denominator_j += tfidf ** 2

        return denominator_i, denominator_j

    # extract idf for calculate similarity between test sentences (idf computes on train corpus)
    def return_idf(self, word):
        if word not in self.idf_score:      # word does not appear in the corpus
            self.miss_idf += 1
            cur_idf_score = 1               # TODO think for a good value to insert
        else:
            cur_idf_score = self.idf_score[word]
            self.hit_idf += 1

        if cur_idf_score < 1:
            cur_idf_score = 1

        return cur_idf_score

    # normalize similarity_matrix and return a markov matrix (each row sum to one)
    def _markov_matrix(self, similarity_matrix):
        '''
        create a mrakov matrix, only normilze sum of row (=1), without applying threshold
        :param similarity_matrix: similarity matrix contain tf-idf-cosine-similarity between each of two edges
        :return: markov matrix
        '''
        row_sum = similarity_matrix.sum(axis=1, keepdims=True)

        return similarity_matrix / row_sum

    ####################################
    # write result into HTML file #
    # ####################################

    # build HTML file with text color regrads to word contribute
    def build_summarization_html(self):
        # create red-green color settings
        self.create_red_green_gradient()

        # directory path
        html_dir_path = self.create_file_path()
        file_path = html_dir_path + str(self.desc_id) + '.html'

        # word cont dict - learn from corpus - self.word_cont_dict
        with open(file_path, 'w') as myFile:
            myFile.write('<html>')
            myFile.write('<body>')

            self.write_html_header(myFile)          # write algorithm meta data into HTML header file
            self.write_sentence_header(myFile)      # write sentence according to their score of LexRank variation score
            self.write_summarization_output(myFile)     #

            myFile.write('</body>')
            myFile.write('</html>')
            myFile.close()
            logging.info('save html file: ' + str(file_path))
        return

    # build html header (Algorithm main configuration) according to method selected
    def write_html_header(self, myFile):

        myFile.write('<p><b>LexRank version: </b>' + str(self.lex_rank_algorithm_version) + '</p>')
        myFile.write('<p><b>Item title: </b>' + str(self.desc_title) + '</p>')

        # header according to LexRank version
        if self.lex_rank_algorithm_version == 'personality-based-LexRank':
            myFile.write('<p><b>User Personality traits: </b>')

            user_personality_text = ', '.join(
                '{}: {}'.format(trait, group) for trait, group in self.user_personality_dict.items()
            )

            myFile.write('<span style="background-color: ' + str('#ffffff') + ';">' + user_personality_text + '</span>')
            myFile.write('</p>')
            myFile.write('<p><b>Damping factor: </b>' + str(self.damping_factor) + ' <b>, Edge threshold: </b>' +
                         str(self.threshold) + '</p>')
            myFile.write('<p><b>Adjacency Matrix=</b>' + str(1 - self.damping_factor) + ' * Personality Matrix + ' +
                         str(self.damping_factor) + ' * Personality-Similarity Matrix' + '</p>')

        elif self.lex_rank_algorithm_version == 'vanilla-LexRank':
            myFile.write('<p><b>Damping factor: </b>' + str(self.damping_factor) + ' <b>, Edge threshold: </b>' +
                         str(self.threshold) + '</p>')
            myFile.write('<p><b>Adjacency Matrix=</b>' + str(1 - self.damping_factor) + ' * Uniform Matrix + ' +
                         str(self.damping_factor) + ' * Similarity Matrix' + '</p>')

        myFile.write('<p><b>Corpus num documents: </b>' + str(self.doc_number_total) +
                     ' <b>, Unigram corpus features: </b>' + str(self.unigram_corpus_features) + '</p>')

    # write meta data about sentences - according to their LexRank score
    def write_sentence_header(self, myFile):

        myFile.write('<p><b>LexRank version </b>' + str(self.lex_rank_algorithm_version) + ' output:</p>')

        # write sentences selected by importance order
        for sen_idx, sentence_str in enumerate(self.summary):
            # sentence_words = self.analyze(sentence_str)  # use original vectorizer to split test sentences data
            myFile.write('<p>')
            sen_real_index = self.sorted_ix[sen_idx]

            sentence_header = '<span style="background-color: ' + str('#ffffff') + ';">' + 'Idx: <b>' + \
                              str(sen_real_index) + '</b>, LexScore: <b>' + \
                              str(round(self.lex_scores[sen_real_index], 3)) + '</b>, '

            # relevant for personality-baesd LexRank
            if sen_real_index < len(self.list_avg_contribute):
                sentence_header += 'Avg personality contribute: <b>' + \
                                   str(round(self.list_avg_contribute[sen_real_index], 2)) + '</b>, '

            sentence_header += 'Words: ' + '</span>'
            myFile.write(sentence_header)

            myFile.write('</p>')
            myFile.write('<p>')
            sentence_words = sentence_str.split(' ')

            # write words with background
            for cur_word in sentence_words:
                token_list = self.analyze(cur_word)
                if len(token_list) > 0:
                    cur_word_token = token_list[0]  # return list - use first words TODO fix
                else:
                    cur_word_token = ''

                if cur_word_token in self.word_cont_dict:
                    cur_word_cont = self.word_cont_dict[cur_word_token]
                    cur_background = self.get_background_color(cur_word_cont)
                    myFile.write('<span style="background-color: ' + str(cur_background) + ';opacity: 0.8;">' +
                                 str(' ') + str(cur_word.encode('utf-8')) + str(' ') +
                                 '</span>')
                else:
                    myFile.write('<span style="background-color: ' + str('#ffffff') + ';">' +
                                 str(' ') + str(cur_word.encode('utf-8')) + str(' ') +
                                 '</span>')
            myFile.write('</p>')

    # write summarization according to summarization version
    def write_summarization_output(self, myFile):
        myFile.write('<p></p><p>')
        for cur_sentence in self.description_summary_list:
            myFile.write('<span>' + str(cur_sentence.encode('utf-8')) + '. <br>' + '</span>')
        myFile.write('</p>')

        # write sentences which discarded due to high similarity
        self.write_sentences_discarded_participate(myFile)

    def write_sentences_discarded_participate(self, myFile):
        dis_str_sentences = ''
        par_str_sentences = ''
        for sen_idx in self.discarded_sentences:
            dis_str_sentences += str(sen_idx)
            dis_str_sentences += ', '
        dis_str_sentences = dis_str_sentences[: -2]

        for sen_idx in self.sentence_already_inserted:
            par_str_sentences += str(sen_idx)
            par_str_sentences += ', '
        par_str_sentences = par_str_sentences[: -2]
        myFile.write(
            '<p><b> Sentences inside: </b> {} <b> Sentence discarded: </b> {} </p>'.format(
                str(par_str_sentences), str(dis_str_sentences))
        )

    # create list of color and percentile settings
    def create_red_green_gradient(self):
        red = Color("red")
        number_gradient_color = 50
        self.color_list = list(red.range_to(Color("green"), number_gradient_color))     # 10 color correspond to percentile values
        self.percentile_list = self.percentile_color(number_gradient_color)               # 10 diff percentile values

    # get percentile of word contribute values
    def percentile_color(self, num_color):
        import numpy as np
        c_l = self.word_cont_dict.values()
        ratio = np.float(100)/np.float(num_color)
        percentile_list = [np.percentile(list(c_l), int(i * ratio)) for i in range(1, num_color+1)]
        logging.info('Percentile contribute list: ' + str(percentile_list))
        return percentile_list

    # determine background color of word regard to personality contribute value
    def get_background_color(self, contribute):
        for per_idx, per_val in enumerate(self.percentile_list):
            if contribute <= per_val:
                hex_color = self.color_list[per_idx]
                return hex_color

    # directory path
    def create_file_path(self):

        html_dir_path = self.html_dir + str(self.cur_time) + '_LexRank_version=' + str(self.lex_rank_algorithm_version)

        if self.lex_rank_algorithm_version == 'personality-based-LexRank':
            personality_str = ''
            for trait, value in self.user_personality_dict.items():
                personality_str += '_' + str(trait[0]) + '=' + str(value)
            html_dir_path += str(personality_str)
        elif self.lex_rank_algorithm_version == 'vanilla-LexRank':
             pass

        html_dir_path = html_dir_path + '_summarization_max_similarity=' + str(self.summarization_similarity_threshold)

        html_dir_path = html_dir_path + '_damping_factor=' + str(self.damping_factor)

        html_dir_path += '/'

        import os
        if not os.path.exists(html_dir_path):
            os.makedirs(html_dir_path)

        logging.info('file dir: {}'.format(str(html_dir_path)))
        return html_dir_path
