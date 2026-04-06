import datetime

def get_time_string():
  return str(datetime.datetime.now())[2:19].replace(' ', '_').replace('-','').replace(':', '').replace('.', '_')