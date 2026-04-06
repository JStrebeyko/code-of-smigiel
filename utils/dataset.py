import pandas as pd
from utils.fileio import get_from_cache, write_to_cache, save_dataset, ensure_raw
from utils.presets import get_process_preset

class Dataset():
    def __init__(self, name):
        print(f"""
Initializng dataset: {name}""")
        self.name = name
        self.raw = ensure_raw(name)
        print(f" raw data: {len(self.raw)}")

        self.preset = get_process_preset(name)
        self.data = self.raw.apply(self.preset.prestep) if (self.preset.prestep and len(self.raw)) else self.raw
        self.prefix = get_from_cache('prefix', name)
        self.prompt = get_from_cache('prompt', name)
        print(f" prestepped: {len(self.data)}") if self.preset.prestep else None


    def extract(self):
        print(" extracting...")

        # depenging whether our data is a Series or DataFrame, extraction works a bit differently
        is_multicolumn = len(self.data.columns) > 1
        print(f"    is raw multicolumn?: {is_multicolumn}")
        self.extracted = [self.preset.prefix(self.data.iloc[[index]]) for index in range(0, len(self.data))] if is_multicolumn else  [self.preset.prefix(item) for item in self.data["text"]]
        df = pd.DataFrame(data={"prefix": self.extracted})
        write_to_cache(df, 'prefix', self.name)
        return df

    def promptify(self):
        print(f"  promptifying {self.name}...")
        prefix = get_from_cache('prefix', self.name)["prefix"]
        self.prompts = [self.preset.prompt(a_prefix) for a_prefix in prefix]
        df = pd.DataFrame(data={"prompt": self.prompts})
        write_to_cache(df, 'prompt', self.name)
        return self.prompts
    
    def update_preprocessed(self):
        print(f"  Updating {self.name}.csv")
        text = self.data["text"]
        prefix = get_from_cache('prefix', self.name)["prefix"]
        prompt = [self.preset.prompt(x) for x in prefix]

        fresh_df = pd.DataFrame(data={"text": text, "prefix": prefix, "prompt": prompt, "dataset": self.name} )
        save_dataset(fresh_df, self.name)
        return fresh_df

