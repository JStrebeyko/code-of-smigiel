from datasets import load_dataset
import os
import csv
import kagglehub
from .consts import CACHE_PATH, PREPROCESSED_PATH, RANDOM_STATE, domain_to_dataset
import pandas as pd
from pathlib import Path

BASE_PER_DOMAIN_AMMOUNT = 12000

def safe_write(data, path):
    the_path_to_write_to = Path(path)
    if not the_path_to_write_to.parent.is_dir():
        the_path_to_write_to.parent.mkdir(parents=True, exist_ok=True)
    # QUOTE_NONNUMERIC ensures bare \r in string values is quoted, preventing
    # the CSV parser from treating it as a row separator on read-back.
    data.to_csv(the_path_to_write_to, index=False, quoting=csv.QUOTE_NONNUMERIC)

def get_data_cache_path(cache_type, cat_name):
   return f"{CACHE_PATH}{cache_type}/{cat_name}.csv"

def get_from_cache(cache_type, cat_name):
  cache_path = get_data_cache_path(cache_type, cat_name)
  if os.path.isfile(cache_path):
      return pd.read_csv(get_data_cache_path(cache_type, cat_name))
  return None

def write_to_cache(data, cache_type, cat_name):
    the_path_to_write_to = Path(get_data_cache_path(cache_type, cat_name))
    safe_write(data, the_path_to_write_to)

def save_dataset(data, cat_name):
   path_to_write = f'{PREPROCESSED_PATH}datasets/{cat_name}.csv'
   safe_write(data, path_to_write)

def save_domain(data, domain_name):
   path_to_write = f'{PREPROCESSED_PATH}domains/{domain_name}.csv'
   safe_write(data, path_to_write)

def load_preprocessed(cat_name):
   try:
      return pd.read_csv(f"{PREPROCESSED_PATH}datasets/{cat_name}.csv")
   except Exception:
      return None


def get_raw_data(dataset_name):
   
   # WIKI
   if dataset_name == 'wiki':
      wiki_raw = pd.DataFrame(data={"text": [text for text in load_dataset("chrisociepa/wikipedia-pl-20230401", split='train').to_pandas()["text"].sample(BASE_PER_DOMAIN_AMMOUNT,replace=False, random_state=RANDOM_STATE)]})
      wiki_raw.reset_index(inplace=True, drop=True)
      write_to_cache(wiki_raw, 'raw', dataset_name)
      return wiki_raw

  # LIT
   elif dataset_name == 'plsc':
      abstracts_df = load_dataset("rafalposwiata/plsc", split="train").to_pandas()
      # data length: 159767
      # filter early
      sampled = abstracts_df.sample(BASE_PER_DOMAIN_AMMOUNT, replace=False, random_state=RANDOM_STATE)
      df = pd.DataFrame(data={"text": sampled["abstract"], "journal": sampled["journal"], "title": sampled["title"]}).reset_index(drop=True)
      write_to_cache(df, 'raw', dataset_name)
      return df
   
   elif dataset_name == 'coursebooks':
      # 1288
      courses_df = load_dataset("rafalposwiata/open-coursebooks-pl", split="train").to_pandas()
      courses_df["num_of_paragraphs"] = courses_df["paragraphs"].apply(len)
      courses_df = courses_df[(courses_df["num_of_paragraphs"] > 0) & ~(courses_df["chapter"].isna()) & ~(courses_df["chapter"] == '')]
      courses_df["text"] = courses_df["paragraphs"].apply(lambda x: " ".join(x))
      courses_df.reset_index(inplace=True, drop=True)
      write_to_cache(courses_df, 'raw', dataset_name)
      return courses_df
  
   elif dataset_name == 'classics':
      cache_path = kagglehub.dataset_download("dmitriilebedev/polish-corpus")
   
      print(cache_path)
      lines = []
      with open(f"{cache_path}/literatura_polska.txt") as file:
         lines = [line.strip() for line in file.readlines() if len(line.strip())]
         # over 27,000 rows - no need for that many, early sampling
         classic_df = pd.DataFrame(data={"text": [line for line in lines if len(line) > 100]})
         print(f"raw classics len: {len(classic_df)}")
         write_to_cache(classic_df, 'raw', dataset_name)
         return classic_df

   # SOCIAL
   elif dataset_name == 'twitter':
      # 35,921 
      text_raw = pd.DataFrame(data={"text": load_dataset("clarin-pl/twitteremo", split='train').to_pandas()["tekst"]})
      write_to_cache(text_raw, 'raw', dataset_name)
      return text_raw
  
   elif dataset_name == 'wykop':
      # BAN-PL dataset: https://github.com/ZILiAT-NASK/BAN-PL
      # Zip files are password-protected; passwords match the zip filenames (per README)
      import requests
      import zipfile
      import io as _io

      ban_files = [
         ("BAN-PL_1.zip", "BAN-PL_1"),
         ("BAN-PL_2.zip", "BAN-PL_2"),
      ]
      base_url = "https://raw.githubusercontent.com/ZILiAT-NASK/BAN-PL/main/data/"

      dfs = []
      for zip_name, password in ban_files:
         url = base_url + zip_name
         print(f"Downloading {url}...")
         response = requests.get(url, timeout=120)
         response.raise_for_status()
         with zipfile.ZipFile(_io.BytesIO(response.content)) as zf:
            zf.setpassword(password.encode())
            csv_name = next(n for n in zf.namelist() if n.endswith('.csv'))
            with zf.open(csv_name) as f:
               dfs.append(pd.read_csv(f))

      both_bans = pd.concat(dfs, ignore_index=True)
      nice_texts = both_bans[both_bans["Class"] == 0]["Text"]
      raw_text = pd.DataFrame(data={"text": nice_texts}).reset_index(drop=True)
      write_to_cache(raw_text, 'raw', dataset_name)
      return raw_text


   # REVIEWS
   elif dataset_name== 'polemo_hotels':
      hotels_raw = load_dataset("clarin-pl/polemo2-official", "hotels_text", split='train+test+validation').to_pandas()["text"]
      hotels_df = pd.DataFrame(data={"text": hotels_raw})
      write_to_cache(hotels_df, "raw", dataset_name)
      return hotels_df

   elif dataset_name== 'polemo_medicine':
      medicine_raw = load_dataset("clarin-pl/polemo2-official", "medicine_text", split='train+test+validation').to_pandas()["text"]
      medicine_df = pd.DataFrame(data={"text": medicine_raw})
      write_to_cache(medicine_df, "raw", dataset_name)
      return medicine_df
   
   elif dataset_name== 'polemo_products':
      products_raw = load_dataset("clarin-pl/polemo2-official", "products_text", split='train+test+validation').to_pandas()["text"]
      products_df = pd.DataFrame(data={"text": products_raw})
      write_to_cache(products_df, "raw", dataset_name)
      return products_df

   elif dataset_name == 'polemo_courses':
      courses_reviews_raw = load_dataset("clarin-pl/polemo2-official", "reviews_text", split='train+test+validation').to_pandas()["text"]
      coursed_df = pd.DataFrame(data={"text": courses_reviews_raw})
      write_to_cache(coursed_df, "raw", dataset_name)
      return coursed_df

   elif dataset_name == 'allegro':
      raw = load_dataset("PL-MTEB/allegro-reviews", split='train+test+validation')["text"]
      # raw: 11585 let's say we let it pass
      df = pd.DataFrame(data={"text": raw})
      write_to_cache(df, "raw", dataset_name)
      return df

   elif dataset_name == 'filmweb':
      # Filmweb+ dataset: https://github.com/narolski/filmwebplus
      # 27202 Polish movie reviews; column 'review' holds text
      import requests
      import gzip
      import io as _io

      url = "https://raw.githubusercontent.com/narolski/filmwebplus/master/filmwebplus.gz"
      print(f"Downloading {url}...")
      response = requests.get(url, timeout=120)
      response.raise_for_status()
      with gzip.open(_io.BytesIO(response.content)) as f:
         all_reviews = pd.read_csv(f)
      sampled = all_reviews.sample(BASE_PER_DOMAIN_AMMOUNT, replace=False, random_state=RANDOM_STATE)
      df = pd.DataFrame(data={"text": sampled["review"]}).reset_index(drop=True)
      write_to_cache(df, "raw", dataset_name)
      return df

   elif dataset_name == 'pmrd':
      url = 'https://raw.githubusercontent.com/kamilsan/polish-movie-reviews-dataset/refs/heads/main/dataset.json'
      raw = pd.Series([review_obj["review"] for review_obj in pd.read_json(url)["reviews"] if len(review_obj["review"]) > 100]).sample(BASE_PER_DOMAIN_AMMOUNT, replace=False, random_state=RANDOM_STATE)
      df = pd.DataFrame(data={"text":raw})
      write_to_cache(df,"raw", dataset_name)
      return df
   
   elif dataset_name == 'wikinews':
      path = f'{CACHE_PATH}raw/wikinews.jsonl'
      df = pd.read_json(path,orient='records', lines=True)
      print(df['text'].str.len().describe())
      print(df.head())
      return pd.DataFrame(data={"text": df["text"]})


def ensure_raw(dataset_name):
   cached = get_from_cache('raw', dataset_name)
   if cached is not None:
      return cached
   return get_raw_data(dataset_name)


def output_domains(domain_list, n, output_file_path):
   """Create the LLM input file by sampling n rows from each domain in domain_list and concatenating them together. Output is saved to output_file_path"""
   if not output_file_path or not output_file_path.endswith('.csv'):
      raise ValueError("No output_file_path CSV provided")
   # Sort by canonical domain order (defined in consts.py) so that pd.concat always
   # produces the same row layout regardless of the order domains were passed on the CLI.
   # This is required for sample(random_state=RANDOM_STATE) to be reproducible:
   # sample draws by positional index, so row layout must be identical across runs.
   canonical_order = list(domain_to_dataset.keys())
   ordered = sorted(domain_list, key=lambda d: canonical_order.index(d) if d in canonical_order else len(canonical_order))
   df = pd.concat([pd.read_csv(f"data/preprocessed/domains/{domain}.csv") for domain in ordered])
   if n == 0:
      n = len(df)
   output = df.sample(n, replace=False, random_state=RANDOM_STATE)
   output.to_csv(output_file_path, index=False)
   print(f"output is {len(output)} rows long, saved to {output_file_path}")
   return output