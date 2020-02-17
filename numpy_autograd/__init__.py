import os 
os.system('jupyter nbconvert --to script np_wrapping.ipynb')
from .np_wrapping import *