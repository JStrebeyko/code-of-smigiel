import pandas as pd
from utils.fileio import save_domain
from itertools import cycle
from utils.consts import domain_to_dataset, RANDOM_STATE

def balance_sample(dfs, cap, random_state=RANDOM_STATE):
    """Rearrange sizes to equally contribute towards total.
  Lacks caused by sizes smaller than equal share are filled
  with (equal) representation from the remaining 
  """
    n_dfs = len(dfs)
    print(f"dfs: {n_dfs}")
    per_df = min([df.shape[0] for df in dfs]) if cap == 0 else cap // n_dfs
    print(f"per: {per_df}")
    remainder = cap % per_df
    print(f"remainder: {remainder}")
    sampled = []

    # Pre-shuffle all DataFrames once
    shuffled_dfs = [
        df.sample(frac=1, random_state=random_state)
        for df in dfs
    ]

    # Step 1: Take per_df rows from each DataFrame
    for df_shuffled in shuffled_dfs:
        print(f"Taking the fair share of {per_df}")
        the_equal_share = df_shuffled.iloc[:per_df]
        print(f"eqal: {len(the_equal_share)}")

        remainder = remainder + max(0, per_df - len(the_equal_share))
        print(f"remainder: {remainder}")

        sampled.append(the_equal_share)

    # Step 2: Distribute the remainder using round-robin
    extra_needed = remainder
    i = 0
    df_cycle = cycle(shuffled_dfs)

    while extra_needed > 0:
        df = next(df_cycle)
        # Attempt to take the (per_df + i)-th row if it exists
        extra_row = df.iloc[per_df + i : per_df + i + 1]
        if not extra_row.empty:
            sampled.append(extra_row)
            extra_needed -= 1
        i += 1
        if i > max(len(df) for df in dfs):  # Avoid infinite loop in edge cases
            break
    
    # print(f"balanded: {list(map(len,sampled))}")
    balanced = pd.concat(sampled, ignore_index=True)
    print(f"BALANCED: {len(balanced)}")
    return balanced


def prepare_domain(domain_name, cap):
    """Prepare a domain by balancing and saving its datasets."""
    domain_dfs = [pd.read_csv(f'data/preprocessed/datasets/{dataset_name}.csv') for dataset_name in domain_to_dataset[domain_name]]
    domain_ds = balance_sample(domain_dfs, cap)

    domain_ds["domain"] = domain_name
    domain_ds.reset_index(inplace=True, drop=True)
    save_domain(domain_ds, domain_name)
    return domain_ds
    
