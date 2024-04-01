import os

def find_dotenv(start_path=None):
    if start_path is None:
        start_path = os.path.dirname(os.path.abspath(__file__))
    current_path = os.path.abspath(start_path)
    while True:
        file_path = os.path.join(current_path, '.env')
        if os.path.isfile(file_path):
            print(f".env file found at {file_path}")
            return file_path
        new_path = os.path.dirname(current_path)
        if current_path == new_path:
            print("Reached the root directory, .env file not found.")
            return None
        current_path = new_path

def load_env():
    dotenv_path = find_dotenv()
    if dotenv_path:
        print(f"Loading .env file from {dotenv_path}")
        with open(dotenv_path) as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                key, value = line.strip().split('=', 1)
                os.environ[key] = value
                print(f"Loaded {key} from .env file")
    else:
        print("No .env file found to load.")
    #return value
