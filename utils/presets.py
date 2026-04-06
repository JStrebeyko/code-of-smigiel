from typing import Callable

from dataclasses import dataclass
from utils.extract import get_begginning, get_abstract_prefix, extract_coursebook_info
from utils.prompt import (
  get_wiki_prompt,
  get_wikinews_prompt,
  get_coursebook_prompt,
  get_abstract_prompt,
  get_classic_lit_prompt,
  get_social_prompt,
  get_review_prompt,
  get_gov_prompt,
)
from utils.consts import (
  LITERATURE_MIN_LINE_LEN,
  SOCIAL_TEXT_CHAR_MIN_LEN,
  MIN_REVIEW_CHAR_LEN,
  MIN_GOV_PREFIX_LEN,
)
from utils.clean import (
  clean_square_brackets_and_their_content,
  standardize_social_ats,
  clean_extra_whitepsaces,
  clean_newlines,
  clean_special_characters,
  remove_greetings,
)
import re

# REPRODUCIBILITY CONTRACT — read before touching random state here.
#
# Social dataset reproducibility (consistent @userXXXX IDs and prefix lengths)
# depends on a chain of three things held constant across runs:
#
#   1. Lambo (the sentence segmenter, imported in extract.py) seeds Python's
#      global `random` module with a fixed value during Lambo.get() at import time.
#
#   2. social_prestep (see below) calls standardize_social_ats() in a double-call
#      pattern — once for the len() filter check, once for the stored value — which
#      consumes a deterministic number of global random calls before extraction runs.
#
#   3. lambo.segment() during prefix extraction then draws from the global RNG at
#      the same point every run, producing identical sentence boundaries and thus
#      identical prefix lengths and @userXXXX assignments.
#
# DO NOT call random.seed(...) here: it overrides lambo's seed (step 1), shifts
# the RNG state seen by lambo in step 3, and breaks both prefix lengths and AT
# reproducibility. This was confirmed experimentally — see git history.
#
# IMPROVEMENT OPPORTUNITY (would break OG reproducibility):
# Replace random.randint in standardize_social_ats with a dedicated
# random.Random(_at_rng) instance seeded independently, and simplify
# social_prestep to a single call per post. This would decouple AT anonymization
# from lambo's RNG entirely, making the system robust to lambo version changes.
# Currently reproducibility is an undocumented side-effect of lambo's fixed seed.


@dataclass
class ProcessPreset:
    prestep: Callable = None
    prefix: Callable = None
    prompt: Callable = None


def social_prestep(posts):
    denewlinezed=[re.sub(r"\n+"," ", post).strip() for post in posts]
    print("SOCIAL PRESTEP")
    print(f"SOCIAL_TEXT_CHAR_MIN_LEN: {SOCIAL_TEXT_CHAR_MIN_LEN}")
    print(f"before {len(posts)}")
    processed = [standardize_social_ats(post) for post in denewlinezed if (len(standardize_social_ats(post)) >= SOCIAL_TEXT_CHAR_MIN_LEN)]
    print(f"after {len(processed)}")
    return processed

social_preset = ProcessPreset(
        prestep=social_prestep,
        prefix= lambda x: get_begginning(x, 26),
        prompt = get_social_prompt,
    )


def get_review_preset(review_type):
    return ProcessPreset(
        prestep= lambda data: [clean_extra_whitepsaces(text) for text in data if (len(clean_extra_whitepsaces(text)) >= MIN_REVIEW_CHAR_LEN)],
        prefix= lambda text: get_begginning(text, 26),
        prompt = lambda prefix: get_review_prompt(prefix, review_type)
    )


def wikinews_prestep(news):
    print("WIKINEWS PRESTEP")
    print(f"SOCIAL_TEXT_CHAR_MIN_LEN: {SOCIAL_TEXT_CHAR_MIN_LEN}")
    print(f"before {len(news)}")
    cleaned = clean_special_characters(news)
    processed = [post for post in cleaned if (len(post)) >= SOCIAL_TEXT_CHAR_MIN_LEN]
    print(f"after {len(processed)}")
    return processed


def gov_prestep(news):
    print("GOV PRESTEP")
    print(f"MIN_GOV_PREFIX_LEN: {MIN_GOV_PREFIX_LEN}")
    print(f"before {len(news)}")
    cleaned = clean_special_characters([remove_greetings(n) for n in news])
    processed = [post for post in cleaned if (len(post)) >= MIN_GOV_PREFIX_LEN]
    print(f"after {len(processed)}")
    return processed



wikinews_preset = ProcessPreset(
    prestep=wikinews_prestep,
    prefix=lambda text: text,
    prompt=get_wikinews_prompt
)

datasets_to_presets = {
    "wiki": ProcessPreset(
        # no prestep was applied, leading to problems
        # the adhoc substitution in hopes of making
        # the wiki prefixes loadable
        prefix = lambda text: get_begginning(text, 50),
        prompt = get_wiki_prompt
        ),

    "wikinews": ProcessPreset(
        prestep= wikinews_prestep,
        prefix= lambda text: get_begginning(text, 26),
        prompt = lambda prefix: get_wikinews_prompt(prefix)
    ),

    # started to implement this separately
    "gov": ProcessPreset(
        prestep = gov_prestep,
        prefix = lambda text: get_begginning(text, MIN_GOV_PREFIX_LEN),
        # prefix = lambda text: text[:SOCIAL_TEXT_CHAR_MIN_LEN],
        prompt = get_gov_prompt
    ),

    # LIT
    "coursebooks": ProcessPreset(
        prefix=extract_coursebook_info,
        prompt= get_coursebook_prompt
        ),

    "plsc": ProcessPreset(
        prestep=clean_newlines,
        prefix=get_abstract_prefix,
        prompt=get_abstract_prompt
        ),

    "classics": ProcessPreset(
        prestep=lambda x: [clean_square_brackets_and_their_content(line) for line in x if len(clean_square_brackets_and_their_content(line)) >= LITERATURE_MIN_LINE_LEN],
        prefix=lambda text: get_begginning(text, LITERATURE_MIN_LINE_LEN),
        prompt=lambda prefix: get_classic_lit_prompt(prefix)
        ),

    # social
    "twitter": social_preset,
    "wykop": social_preset,

    # reviews
    "polemo_hotels": get_review_preset("hotels"),
    "polemo_medicine":get_review_preset("medicine"),
    "polemo_products": get_review_preset("products"),
    "polemo_courses":get_review_preset("courses"),
    "allegro": get_review_preset("products"),
    "filmweb":get_review_preset("movies"),
    "pmrd":get_review_preset("movies"),
}

def get_process_preset(dataset_name):
    return datasets_to_presets[dataset_name]
