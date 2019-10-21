test_lexrank = {
    'summary_size': 'max',                          # summarization length - 'max' or int
    'HTML_summary_size': 3,
    'threshold': 0.03,                              # min edge weight between two sentences - below remove the edge
    'damping_factor': 0.1,
    'summarization_similarity_threshold': 0.55,

    'personality_trait_dict': {
        'openness': 'H',
        'conscientiousness': 'H',
        'extraversion': 'H',
        'agreeableness': 'H',
        'neuroticism': 'H'
    },
    'lex_rank_algorithm_version': 'personality-based-LexRank',      # 'vanilla-LexRank', 'personality-based-LexRank'
    'products_ids': ['B0746GQ56P'],

    'corpus_size': 50,            # 'max'/int - the amonut of description for IDF calculation- high number is leading to high computation time

    # please don't change (damping factor:1 same effect)
    'personality_word_flag': True,
    'random_walk_flag': True,                        # flag if combine random jump between sentences
    'corpus_path_file': '../data/merge_20048.csv',          # calculate idf from
    'target_item_description_file': '../data/product_description.csv',

    'trait_relative_path_dict': '../data/all_words_contribute',
}