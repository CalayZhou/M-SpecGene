from importlib import import_module
module = import_module(f'mmseg.utils')
module.register_all_modules(False)