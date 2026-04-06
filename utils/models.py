from transformers import AutoTokenizer, AutoModelForCausalLM, Gemma3ForCausalLM
import torch
from dotenv import load_dotenv
from huggingface_hub import login
import os

from utils.consts import MONIKIER_TO_HF_ID_MAP

def hf_login():
    load_dotenv()
    token = os.getenv('HF_TOKEN')
    login(token)

class GeneralModel():
  def __init__(self, name):
    self.name = name


class HF_Model():
  def __init__(self):
    hf_login()
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")


class Auto_HF_Model_With_Chat_Template_Returning_Dict(HF_Model):
  """Mistral, bielik and llama
  """
  def __init__(self, model_id):
    self.model_id = model_id
    print(f'hello, this is {self.model_id}')
    self.processor = AutoTokenizer.from_pretrained(self.model_id, device_map="auto")
    self.model = AutoModelForCausalLM.from_pretrained(
        self.model_id,
        pad_token_id=self.processor.eos_token_id,
        torch_dtype=torch.bfloat16,
        # low_cpu_mem_usage=True,
        device_map='auto',
    )
    print(f"model hf device map: {self.model.hf_device_map}")
    for name, param in self.model.named_parameters():
      if param.is_meta:
          print(f"{name} is on meta (not initialized).")

  def tokenize(self, text, **kwargs):
    messages = [
        {"role": "user", "content": text},
    ]

    tokenized_chat = self.processor.apply_chat_template(
      messages,
      tokenize=True,
      add_generation_prompt=True,
      return_tensors="pt",
      # return_dict=True, # yup, this is the only small difference
      **kwargs
    ).to(self.model.device)

    return tokenized_chat
  
  def infer(self, text, **generation_parameters):

    tokenized_chat = self.tokenize(text, return_dict=True)
    print(text)

    input_len = tokenized_chat["input_ids"].shape[-1]

    # plum does not like being passed a ['token_type_ids'] (note: typos in the generate arguments will also show up in this list)
    if not self.model_id.startswith('CYFRAGOVPL') and 'token_type_ids' in tokenized_chat:
      generation_parameters["token_type_ids"] = tokenized_chat["token_type_ids"]

    generated_ids = self.model.generate(
        input_ids = tokenized_chat["input_ids"],
        attention_mask=tokenized_chat["attention_mask"],
        pad_token_id=self.processor.eos_token_id,
        **generation_parameters
    ).to(self.model.device)

    generated_ids = generated_ids[0][input_len:]


    decoded = self.processor.decode(generated_ids, skip_special_tokens=True)
    return decoded


class Gemma(HF_Model):
  """Gemma 3 from Google
  https://huggingface.co/google/gemma-3-27b-pt
  """
  def __init__(self):
    self.model_id = "google/gemma-3-27b-it"
    print(f'hello, this is Gemma 3, also known as {self.model_id}')

    # Gemma3ForCausalLM for loading Gemma as a text-only model
    # https://huggingface.co/blog/gemma3#detailed-inference-with-transformers
    self.model = Gemma3ForCausalLM.from_pretrained(
        self.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).eval()

    # try processor?
    self.processor = AutoTokenizer.from_pretrained(
      self.model_id,
      )

  def tokenize(self, text, **kwargs):
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": text}]
        }
    ]
    tokenized_chat = self.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        use_fast=True, # otherwise, slow image processor is used
        **kwargs,
    ).to(self.model.device)

    return tokenized_chat
  
  def infer(self, prompt, **generation_parameters):

    tokenized_chat = self.tokenize(prompt,  padding="max_length", return_dict=True)
    input_len = tokenized_chat["input_ids"].shape[-1]

    with torch.inference_mode():
      generation = self.model.generate(**tokenized_chat, **generation_parameters, 
        cache_implementation="static")
      generation = generation[0][input_len:]

    decoded = self.processor.decode(generation, skip_special_tokens=True)
    return decoded


def get_model(monikier):
  if monikier == 'gemma':
    return Gemma()
  id = MONIKIER_TO_HF_ID_MAP[monikier]
  return Auto_HF_Model_With_Chat_Template_Returning_Dict(id)




