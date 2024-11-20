import time
import timeit
import os

import json
import sys
from extract_text import *
from files_operations import *
from json_functions import *
from SQLiteManager import *
parent_dir = os.path.dirname(os.path.abspath(__file__))
mde_root = os.path.dirname(os.path.dirname(parent_dir))

# Add the path to the PatternDetection directory to the sys.path list
pattern_detection_path = os.path.join(mde_root, 'PatternDetection')
sys.path.append(pattern_detection_path)
from pattern_detection_v001 import ImageMatcher