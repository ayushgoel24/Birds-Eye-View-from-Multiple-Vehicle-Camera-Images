import importlib
import os
import sys

class PathUtil:

    @staticmethod
    def get_absolute_path(path):
        return os.path.abspath( os.path.expanduser(path) )
    
    @staticmethod
    def load_module(module_file):
        name = os.path.splitext( os.path.basename(module_file) )[0]
        dir = os.path.dirname(module_file)
        sys.path.append(dir)
        spec = importlib.util.spec_from_file_location(name, module_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module