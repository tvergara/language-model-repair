import sys
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, "memit"))


from .memit.memit import apply_memit_to_model

def memit(model):
    return apply_memit_to_model
