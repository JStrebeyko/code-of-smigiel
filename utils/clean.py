import re
import random

# IMPROVEMENT OPPORTUNITY (would break OG reproducibility):
# Replace random.randint below with a dedicated seeded Random instance, e.g.:
#   from .consts import RANDOM_STATE
#   _at_rng = random.Random(RANDOM_STATE)
# and call _at_rng.randint(...) instead of random.randint(...).
# This would isolate AT anonymization from the global RNG, preventing it from
# shifting the random state that lambo depends on for sentence segmentation.
# Currently, the global random consumption in social_prestep (including the
# double-call pattern — see presets.py) is load-bearing for OG reproducibility.

def format_quotes(some_text):
  """In some categories, like social, special characters are padded with whitespaces
  on both sides. This helper helps to address the problem of formatting quotes, with
  which is is not clear from which side should the extra padding be reduced."""
  pl_chars = "AaĄąBbCcĆćDdEeĘęFfGgHhIiJjKkLlŁłMmNnŃńOoÓóPpRrSsŚśTtUuWwYyZzŹźŻż"
  search_string = fr"(\"[\s[{pl_chars}]+\")"
  formated_quotes = re.sub(search_string, clip, some_text)
  return formated_quotes

def clean_extra_whitepsaces(some_text:str):
  formated_quotes = format_quotes(some_text)
  
  return (formated_quotes
      # crude. replace accepts regexp btw
    .replace(' . . .', '...')
    .replace(' . .', '..')
    .replace(' .', '.')
    .replace(' ,', ',')
    .replace(' :', ':') # emojis oh no...
    .replace('( ', '(')
    .replace(' )', ')')
    .replace('[ ', '[')
    .replace(' ]', ']')
    .replace(' !', '!')
    .replace(' ?', '?')
    .replace(' /', '/')

    .replace(" m ", "m ")
    .replace(" śmy", "śmy")
    .replace(" em ", "em ")
    .replace(' [/ b]', '') # a weird artifact in polemo_courses
          ).strip()


def clip(some_string):
  our_match = some_string.group()
  return f"\"{our_match[2:-2]}\""

def clean_square_brackets_and_their_content(some_string):
    # removing the inline refences present in some texts - like [12], [14], [1155]
    stripped_before = some_string.strip()
    replaced = re.sub(r"\[\d+\]", '', stripped_before)
    return replaced.strip()

# lit_clean('[1] [10]  [d]  [129102] [11]')


# wiki
def clean_newlines(some_strings):
  return [string.replace("\n"," ") for string in some_strings]

# def clean_abstract_newlines(some_strings):

def standardize_social_ats(some_string):
  ad_hoc_user_uuid = f"@user{random.randint(1000, 9999)}"
  # singlify and replace
  twitter_singlified = re.sub(r'(@anonymized_account )+', '@anonymized_account ', some_string)
  twitter_wykop_singlified = re.sub(r'({USERNAME} )+', '{USERNAME} ', twitter_singlified)
  unified = re.sub(r'@anonymized_account|{USERNAME}', ad_hoc_user_uuid, twitter_wykop_singlified)
  return unified.strip()


def clean_special_characters(some_texts):
  LINE_SEPARATOR = "\u2028"
  PARAGRAPH_SEPARATOR = "\u2029"
  return [text.replace('\t',' ')
    .replace('␤','\n')
    .replace(LINE_SEPARATOR, '\n')
    .replace(PARAGRAPH_SEPARATOR, '\n')
    .replace('\r', '')
    # .replace('\n', ' ') # should it remove newlines? ❌
      for text in some_texts]



greetings_to_remove = [
                       'Szanowny Panie Marszałku!',
                       'Panie Marszałku!',
                       'Szanowna Pani Marszałek!',
                       'Szanowny Panie Ministrze!',
                       'Szanowny Pani Ministrze!',
                       'Pani Marszałek!',
                       'Szanowne Panie Posłanki i Panowie Posłowie!',
                       'Pani Minister!',
                       'Wielce Szanowna Pani Marszałkini',
                       'Pani Marszałku!',
                       'Pani Premierze!',
                       'Pani Poseł!',
                       'Panie Rzeczniku!',
                       'Szanowni Obywatele!',
                       'Panie Marszałku!',
                       'Drodzy Polacy!',
                       'Panie Ministrze!',
                       'Panie Komendancie!',
                       'Czcigodny Panie Marszałku!',
                       'Panie i Panowie Posłowie!',
                       'Państwo Ministrowie!',
                       'Panie Posłanki!',
                       'Panowie Posłowie!',
                       'Panowie Posłowie!',
                       'Szanowni Państwo!',
                       'Panie Generale!',
                       'Drodzy Rolnicy!',
                       'Wielce Szanowni Państwo!',
                       'Szanowna Pani Minister!',
                       'Wysoka Izbo!',
                       'Panowie Generałowie!',
                       'Szanowny Panie Premierze!',
                       'Panie Premierze!',
                       'Szanowni Państwo Ministrowie!',
                       'Szanowna Pani Marszałkini!',
                       'Pani Marszałkini!',
                       'Wysoka Izba!',
                       'Panowie Oficerowie!',
                       'Panowie Ministrowie!',
                       'Panie Posłanki i Panowie Posłowie!',
                       'Panie Prezesie!',
                       'Państwo Komendanci!',
                       "Szanowni Przedstawiciele Inicjatywy Ustawodawczej!",
                       'Wysoki Sejmie!']


def remove_greetings(text: str, greetings=greetings_to_remove) -> str:
    """
    Removes all consecutive greetings from the beginning of a string.
    """
    # one greeting with optional surrounding whitespace
    single = r"(?:\s*(?:" + "|".join(re.escape(g) for g in greetings) + r")\s*)"
    # repeat the whole unit
    pattern = r"^" + single + "+"
    return re.sub(pattern, "", text)

# from postprocessing:

                                        # text_file.write(text
                                        #                 .replace('\t',' ')
                                        #                 .replace('␤','\n')
                                        #                 .replace(LINE_SEPARATOR, '\n')
                                        #                 .replace(PARAGRAPH_SEPARATOR, '\n')
                                        #                 .replace('\r', '')
                                        #                 .replace('\n', ' ') + '\n')
