LITERATURE_MIN_LINE_LEN = 100
SOCIAL_TEXT_CHAR_MIN_LEN = 100
MIN_REVIEW_CHAR_LEN = 100
MIN_GOV_PREFIX_LEN = 150
MAX_BASELINE_TOKEN_LEN = 512

RANDOM_STATE = 123123123

# PATHS
CORE_PATH = 'data/'
CACHE_PATH = f'{CORE_PATH}.cache/'
RAW_PATH =f"{CORE_PATH}raw/"
PREPROCESSED_PATH =f"{CORE_PATH}preprocessed/"

# MODELS
MONIKIER_TO_HF_ID_MAP = {
    'llama-sm': 'meta-llama/Llama-3.1-8B-Instruct',
    'bielik-sm': 'speakleash/Bielik-7B-Instruct-v0.1',
    'mistral-sm': 'mistralai/Mistral-7B-Instruct-v0.3',
    
    'plum': 'CYFRAGOVPL/PLLuM-12B-nc-chat',
    'bielik-md': 'speakleash/Bielik-11B-v2.3-Instruct', 
    'mistral-md': 'mistralai/Mistral-Nemo-Instruct-2407',

    'gemma': 'google/gemma-3-27b-it',
    'llama-lg': "meta-llama/Llama-3.3-70B-Instruct",
    ###
    # 'qwen':"Qwen/Qwen3-32B",
    # 'gpt': "openai/gpt-oss-20b",
    ## baselines:
    'bert': 'bert-base-uncased',
}

MULTIMODAL_CONTENT_MODELS = ['gemma', 'llama-sm' 'llama-lg', 'qwen'] # models that can accept images as input, so their chat template expects a list for content

models = MONIKIER_TO_HF_ID_MAP.keys()

# DOMAINS
domain_to_dataset = {
  'wiki': ['wiki'],
  'lit': ['plsc', 'coursebooks', 'classics'],
  'social': ['twitter', 'wykop'],
  'reviews':['polemo_hotels',
             'polemo_medicine',
             'polemo_courses',
             'polemo_products',
             'allegro',
             'filmweb',
             'pmrd'],
  'gamma': ['wikinews',
             'gov'
             ],
}
all_dataset_names = " ".join([" ".join(domain_ds) for domain_ds in domain_to_dataset.values()]).split(" ")

domains = domain_to_dataset.keys()


# STRATEGIES
strategy_to_generate_kwargs = {
    "greedy": {"do_sample": False},
    "sampling": {"do_sample": True, "num_beams": 1},
    "beam_search": {"num_beams": 2},
    "contrastive": {"penalty_alpha":0.6, "top_k":4, "trust_remote_code":True},
    "dbs": {"num_beams":6, "num_beam_groups":3, "diversity_penalty":1.0, "do_sample": False}, # diverse_beam_search
    "llama_plum_sampling": {"do_sample": True, "temperature": 0.6, "top_p": 0.9}
}

dataset_name_to_path = {
  'wikinews': 'data/.cache/raw/wikinews.jsonl'
}

strategies = strategy_to_generate_kwargs.keys()