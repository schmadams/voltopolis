import os
from dotenv import load_dotenv, dotenv_values

def check_env():
    load_dotenv()

if __name__ == '__main__':
    check_env()
    key = os.environ.get("ENTSOE_API_KEY")
    a=1
    a=1