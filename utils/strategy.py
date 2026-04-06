from utils.consts import strategy_to_generate_kwargs

def resolve_strategy_kwargs(strategy):
  print(f"STRATEGY: {strategy}")
  return strategy_to_generate_kwargs[strategy]