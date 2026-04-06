from lambo.segmenter.lambo import Lambo

lambo = Lambo.get('Polish', with_splitter=False)

def parse_to_sentences(text: str):
  document = lambo.segment(text)
  sentences = []
  for turn in document.turns:
    for sentence in turn.sentences:
      sentences.append(sentence.text)
  return sentences

def get_begginning(input_string: str, min_length: int= 1):
  """get prefix"""
  # get only a subset, for very long texts
  subset = input_string[:1000]

  sentences = parse_to_sentences(subset)
  prefix=''
  for sentence in sentences:
    prefix += sentence
    if len(prefix) >= min_length:
      return prefix.strip()
  return prefix.strip()

def get_abstract_prefix(x):
  return f"\"{x['title'].values[0]}\", opublikowanego w piśmie {x['journal'].values[0]}"

def extract_coursebook_info(data):
    coursebook_name = data['coursebook'].values[0]
    chapter = f" {data['chapter'].values[0]}" if len(data['chapter']) else ''
    subject = data['subject'].values[0]
    extracts = f"\"{coursebook_name}\",{chapter} temat: {subject}"
    return extracts

