import copy
import pathlib
import csv
import re
import sys
import os
import random
import hashlib
import time
from langdetect import detect, LangDetectException, DetectorFactory

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from utils.consts import RANDOM_STATE
from utils.clean import remove_greetings

csv.field_size_limit(sys.maxsize)
DetectorFactory.seed = RANDOM_STATE

def mdsum(s):
    return int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16)

INTRODUCTION_LENGTH = 70
REMOVE_OVERLAP_FROM_GENERATIONS = True
REMOVE_OVERLAP_FROM_HUMANTEXT = True
REMOVE_PREFIX_FROM_HUMANTEXT = True
REMOVE_PREFIX_FROM_GENERATIONS = True
MINIMUM_TEXT_LENGTH = 70
MINIMUM_REPETITIONS = 3

UNDESIRED_LINE_OPENINGS_TUPLE = (
    "Rozumiem, że chcesz",
    "Dobrze",
    "Jasne",
    "Oczywiście",
    "Przepraszam",
    "Przykro mi",
    "Dziękuję",
    "Nie mogę",
    "Nie jestem w stanie",
    "Nie jestem pewien",
    "Nie rozumiem",
    "Jestem tutaj, aby pomóc",
    "Witaj",
    "Proszę bardzo, oto ciąg",
    "Wyjaśnienie stylu",
    "Oczywiście, jeśli masz jakieś uwagi lub sugestie",
    "Czy chciałbyś/chciałabyś, żebym",
    "Oto dalszy ciąg",
    "Wypowiedź polskiego parlamentarzysty",
)

# Maps every known model name to its tier (1=small, 2=medium, 3=large).
# Both spellings of the small Llama are listed; whichever appears in the
# models list is used directly.
MODEL_TIERS = {
    'bielik-sm':   1,
    'mistral-sm':  1,
    'small-llama': 1,
    'llama-sm':    1,
    'bielik-md':   2,
    'mistral-md':  2,
    'plum':        2,
    'gemma':       3,
    'llama-lg':    3,
}

LINE_SEPARATOR = "\u2028"
PARAGRAPH_SEPARATOR = "\u2029"


def _clean(text, replace_tabs=True):
    result = text.replace('\t', ' ') if replace_tabs else text
    return (result
            .replace('␤', '\n')
            .replace(LINE_SEPARATOR, '\n')
            .replace(PARAGRAPH_SEPARATOR, '\n')
            .replace('\r', '')
            .replace('\n', ' '))


def load_generations(input_paths, models, col_offset=0):
    """Read CSVs, group by mdsum(human_text), return (generations, metadata)."""
    models_to_id = {models[i]: (i + 1) for i in range(len(models))}
    generations = {}
    metadata = {}
    for path in input_paths:
        path = pathlib.Path(path)
        with path.open(newline='') as file:
            csvreader = csv.reader(file, delimiter=',')
            for index, row in enumerate(csvreader):
                if not index:
                    print(row)
                    continue
                if row[0] in ('Unnamed: 0', 'Unnamed: 0.1', ''):
                    print("Header, skipping...")
                    continue

                human_text = row[1]
                idd = mdsum(human_text)
                prefix = row[2]
                strategy = row[7 + col_offset]
                generation = row[8 + col_offset]
                modelname = row[9 + col_offset]

                if idd not in generations:
                    generations[idd] = [''] * (len(models) + 1)
                    generations[idd][0] = human_text
                    metadata[idd] = {'strategy': {}}
                assert generations[idd][0] == human_text

                if modelname not in models_to_id:
                    continue
                generations[idd][models_to_id[modelname]] = generation
                metadata[idd]['prefix'] = prefix
                metadata[idd]['strategy'][modelname] = strategy
    return generations, metadata


def filter_generations(generations, metadata, use_remove_greetings=True,
                       use_undesired_filter=True, discard_mid_sentence=True,
                       undesired_phrases=None):
    """Apply all text cleaning/filtering in-place. Returns original (pre-filter) copy.

    undesired_phrases: tuple/list of line-opening strings to drop.
    Defaults to UNDESIRED_LINE_OPENINGS_TUPLE if None.
    """
    if undesired_phrases is None:
        undesired_phrases = UNDESIRED_LINE_OPENINGS_TUPLE
    original = copy.deepcopy(generations)
    print(f"number of generations: {len(original)}")
    for counter, idd in enumerate(generations):
        if counter % 100 == 0:
            print(f"{counter}/{len(original)}")
        for i, text in enumerate(generations[idd]):
            newtext = text.lstrip(";:.,-\u2014\u2013'\"\u201c\u201d\u2018\u2019# ")
            if newtext.startswith(metadata[idd]['prefix']) and (
                    (REMOVE_PREFIX_FROM_HUMANTEXT and i == 0) or (REMOVE_PREFIX_FROM_GENERATIONS and i != 0)):
                if i != 0 or (len(metadata[idd]['prefix']) < 0.5 * len(newtext)):
                    newtext = newtext[len(metadata[idd]['prefix']):]
            newtext = re.sub(r'(\n\s*)+', '\n', newtext).strip()
            firstnewline = newtext.find('\n')
            if firstnewline != -1 and firstnewline != 0 and newtext[firstnewline - 1] == ':' and firstnewline < INTRODUCTION_LENGTH:
                newtext = newtext[firstnewline + 1:]
            overlap = ''
            if i != 0 and REMOVE_OVERLAP_FROM_GENERATIONS:
                overlap = os.path.commonprefix([newtext, generations[idd][0]])
            if i == 0 and REMOVE_OVERLAP_FROM_HUMANTEXT:
                overlap = ''
                for othertext in generations[idd][1:]:
                    prefixhere = os.path.commonprefix([newtext, othertext])
                    if len(prefixhere) != len(overlap):
                        overlap = prefixhere
            if overlap != '':
                newtext = newtext[len(overlap):].lstrip()
                if len(newtext) > 0 and not newtext[0].isupper():
                    m = re.search("\\. [A-Z]", newtext)
                    if m:
                        newtext = newtext[(m.end() - 1):]
                    else:
                        if discard_mid_sentence:
                            newtext = ''
            try:
                if len(newtext) > 10 and detect(newtext) != 'pl':
                    newtext = ''
            except Exception as e:
                print(e)
                print(f"offending string: {newtext}")
                newtext = ''
            try:
                newlines = []
                for line in newtext.split('\n'):
                    cleaned = remove_greetings(line) if use_remove_greetings else line
                    if len(cleaned) > 0 and cleaned[0] == '(' and cleaned[-1] == ')':
                        pass
                    elif len(cleaned) > 10 and cleaned[0] == '(' and detect(cleaned) == 'en':
                        pass
                    elif use_undesired_filter and cleaned.startswith(tuple(undesired_phrases)):
                        pass
                    else:
                        newlines.append(cleaned)
                newtext = '\n'.join(newlines)
            except LangDetectException:
                pass
            newtext = newtext.replace('\n', ' ')
            if len(newtext) > 10:
                inverted = newtext[::-1]
                for ii in range(int(len(newtext) / MINIMUM_REPETITIONS)):
                    length = ii + 1
                    if length > 2 and inverted.startswith(inverted[:length] * MINIMUM_REPETITIONS):
                        repeating_str = inverted[:length]
                        max_times = 0
                        for jj in range(int(len(inverted) / len(repeating_str))):
                            if inverted.startswith(repeating_str * jj):
                                max_times = jj
                            else:
                                break
                        inverted = inverted[(max_times * len(repeating_str)):]
                        newtext = inverted[::-1]
                        break
            if "model językowy" in newtext or newtext.startswith('Przepraszam') or newtext.startswith(
                    'Przykro mi') or newtext.startswith('Dziękuję') or newtext.startswith('Nie mogę'):
                newtext = ''
            generations[idd][i] = newtext.strip()

    # Remove insufficient texts
    for idd in generations:
        for i, text in enumerate(generations[idd]):
            if len(text) < MINIMUM_TEXT_LENGTH:
                generations[idd][i] = ''
    return original


def select_outputs(generations, models, balance_tiers, rng):
    """Tier-balanced random selection → list of (idd, source, text).

    Tiers are derived automatically from the models list via MODEL_TIERS
    (1=small, 2=medium, 3=large).
    """
    models_to_id = {models[i]: (i + 1) for i in range(len(models))}
    def tier_fn(names):
        return [models_to_id[m] for m in names if m in models_to_id]

    def _auto_tier(n):
        return [m for m in models if MODEL_TIERS.get(m) == n]

    tier1 = _auto_tier(1)
    tier2 = _auto_tier(2)
    tier3 = _auto_tier(3)

    acceptable = [0] * (len(models) + 1)
    outputs = []
    for idd in sorted(generations.keys()):
        available_generations = [i for i in range(len(generations[idd])) if not (i == 0 or generations[idd][i] == '')]
        available_humans = [0] if generations[idd][0] != '' else []
        for nbr in available_humans + available_generations:
            acceptable[nbr] += 1
        if available_humans == [] or available_generations == []:
            continue
        x = rng.random()
        if balance_tiers:
            if x < .33:
                available_generations = [i for i in available_generations if i in tier_fn(tier1)]
            elif x < .67:
                available_generations = [i for i in available_generations if i in tier_fn(tier2)]
            else:
                available_generations = [i for i in available_generations if i in tier_fn(tier3)]
        else:
            available_generations = [i for i in available_generations if i in [models_to_id[m] for m in models]]
        if len(available_generations) == 0:
            continue
        selected_generation = rng.choice(available_generations)
        selected_human = 0
        length = min(len(generations[idd][selected_generation]), len(generations[idd][selected_human])) * 0.95
        if rng.random() > 0.5:
            outputs.append((idd, 'human', generations[idd][selected_human][0:int(length)]))
        else:
            outputs.append((idd, models[selected_generation - 1], generations[idd][selected_generation][0:int(length)]))

    rng.shuffle(outputs)
    print(f"len of postprocessed results: {len(outputs)}")
    print("Found acceptable generations:")
    print('\n'.join([(['human'] + models)[i] + '\t' + str(int(100 * acceptable[i] / len(generations))) + '%'
                     for i in range(len(models) + 1)]))
    return outputs


def save_outputs(outputs, metadata, output_path, split=False, split_by_idd=False,
                 replace_tabs=True):
    """Write start.txt/key/meta, optionally into training/test/dev subdirs.

    split_by_idd=True: use idd % 10 for split assignment (hash-based, reproduces
    the Aug-20 base run where dev/test counts differ).
    split_by_idd=False (default): use sequential idx % 10 (uniform split).
    replace_tabs=False: preserve literal tab characters in text output
    (matches dbc0580 base run; tab replacement was added in 1a5635f Aug-20).
    """
    output_path = pathlib.Path(output_path)

    def _write_triple(text_file, key_file, meta_file, idd, source, text):
        found = {}
        if LINE_SEPARATOR in text:
            found["Line Separator (U+2028)"] = text.count(LINE_SEPARATOR)
        if PARAGRAPH_SEPARATOR in text:
            found["Paragraph Separator (U+2029)"] = text.count(PARAGRAPH_SEPARATOR)
        if found:
            print("Unusual line terminators detected:")
            for k, v in found.items():
                print(f"  {k}: {v} occurrence(s)")
        text_file.write(_clean(text, replace_tabs=replace_tabs) + '\n')
        key_file.write('0\n' if source == 'human' else '1\n')
        meta_file.write(str(idd) + '\t' + source + '\t' + (
            'human' if source == 'human' else metadata[idd]['strategy'][source]) + '\n')

    if not split:
        print("NO SPLIT")
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / 'start.txt', "w") as tf, \
             open(output_path / 'start.key', "w") as kf, \
             open(output_path / 'start.meta', "w") as mf:
            for idd, source, text in outputs:
                _write_triple(tf, kf, mf, idd, source, text)
    else:
        for subdir in ('training', 'test', 'dev'):
            (output_path / subdir).mkdir(parents=True, exist_ok=True)
        files = {}
        handles = []
        for split_name in ('training', 'test', 'dev'):
            d = output_path / split_name
            t = open(d / 'start.txt', 'w')
            k = open(d / 'start.key', 'w')
            m = open(d / 'start.meta', 'w')
            files[split_name] = (t, k, m)
            handles.extend([t, k, m])
        try:
            for idx, (idd, source, text) in enumerate(outputs):
                split_key = idd % 10 if split_by_idd else idx % 10
                if split_key == 0:
                    split_name = 'dev'
                elif split_key == 1:
                    split_name = 'test'
                else:
                    split_name = 'training'
                tf, kf, mf = files[split_name]
                _write_triple(tf, kf, mf, idd, source, text)
        finally:
            for h in handles:
                h.close()


def run_postprocess(config: dict):
    """
    Config keys:
      input_paths          list of Path/str
      models               list of model name strings (order affects models_to_id → RNG)
      output_path          Path/str
      balance_tiers        bool
      split                bool (True → training/test/dev subdirs)
      split_by_idd         bool (True → use idd%10 for split; False → idx%10; default False)
      use_remove_greetings bool  (default True)
      use_undesired_filter bool  (default True)
      discard_mid_sentence bool  (default True)
      undesired_phrases    tuple of line-opening strings to drop (default: UNDESIRED_LINE_OPENINGS_TUPLE)
      col_offset           int (default 0; set to 1 for beta CSVs with extra hash col)
      random_state         int (default RANDOM_STATE)
    """
    rs = config.get('random_state', RANDOM_STATE)
    rng = random.Random(rs)
    col_offset = config.get('col_offset', 0)

    print("READING")
    generations, metadata = load_generations(
        config['input_paths'], config['models'], col_offset,
    )

    print("FILTERING")
    filter_generations(
        generations, metadata,
        use_remove_greetings=config.get('use_remove_greetings', True),
        use_undesired_filter=config.get('use_undesired_filter', True),
        discard_mid_sentence=config.get('discard_mid_sentence', True),
        undesired_phrases=config.get('undesired_phrases'),
    )

    print("SELECTING")

    outputs = select_outputs(
        generations, config['models'], config['balance_tiers'], rng,
    )

    print("SAVING")
    save_outputs(outputs, metadata, config['output_path'],
                 split=config.get('split', False),
                 split_by_idd=config.get('split_by_idd', False),
                 replace_tabs=config.get('replace_tabs', True))


if __name__ == '__main__':
    _start_time = time.time()

    # Paths are relative to the project root (one level up from postprocessing/).
    # Batch directories are created by `cli.py generate` as data/{timestamp}_{name}/.
    # Set each BATCH_PATH to the directory that holds the cluster output CSVs for
    # that generation run; input_paths will be all *.csv files inside it.
    DATA_PATH = pathlib.Path(__file__).parent.parent / 'data'

    MODELS_BASE = ['bielik-md', 'bielik-sm', 'small-llama', 'mistral-md', 'mistral-sm', 'plum', 'gemma']

    BASE_BATCH = DATA_PATH / '<YYYYMMDD_HHMMSS_base>'       # ← set to your batch dir
    base_config = {
        'input_paths': sorted(BASE_BATCH.glob('*.csv')),
        'models': MODELS_BASE,
        'output_path': DATA_PATH / 'postprocessed',
        'balance_tiers': True,
        'split': True,
        'split_by_idd': True,          # hash-based split: idd % 10 (not idx % 10)
        'use_remove_greetings': False,
        'use_undesired_filter': False,
        'discard_mid_sentence': False,
        'replace_tabs': False,
    }

    BETA_BATCH = DATA_PATH / '<YYYYMMDD_HHMMSS_beta>'       # ← set to your batch dir
    beta_config = {
        'input_paths': sorted(BETA_BATCH.glob('*.csv')),
        'models': ['llama-lg'],
        'output_path': DATA_PATH / 'postprocessed_beta',
        'balance_tiers': False,
        'split': False,
        'use_remove_greetings': False,
        'use_undesired_filter': False,
        'discard_mid_sentence': False,
        'replace_tabs': True,
    }

    MODELS_GAMMA = ['llama-lg', 'plum', 'gemma', 'mistral-md', 'bielik-md', 'llama-sm', 'mistral-sm', 'bielik-sm']

    GAMMA_BATCH = DATA_PATH / '<YYYYMMDD_HHMMSS_gamma>'     # ← set to your batch dir
    gamma_config = {
        'input_paths': sorted(GAMMA_BATCH.glob('*.csv')),
        'models': MODELS_GAMMA,
        'output_path': DATA_PATH / 'postprocessed_gamma',
        'balance_tiers': True,
        'split': False,
        'use_remove_greetings': True,
        'use_undesired_filter': True,
        'discard_mid_sentence': True,
        'replace_tabs': True,
    }

    config = gamma_config

    run_postprocess(config)

    elapsed = time.time() - _start_time
    print(f"END — total time: {elapsed // 60:.0f}m {elapsed % 60:.1f}s")
