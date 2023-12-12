import yaml
from threading import Lock

class DatasetConfigurationManager:

    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DatasetConfigurationManager, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_file):
        if not self._initialized:
            self.config_file = config_file
            self.load_config()
            self._initialized = True

    def load_config(self):
        with open(self.config_file, 'r') as file:
            self.config = yaml.safe_load(file)