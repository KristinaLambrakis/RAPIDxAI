import os

from path_utils import project_root
import pandas as pd

test_cases = pd.read_csv(os.path.join(project_root, 'test/v3/test_cases.csv'))

test_cases = [dict(c) for _, c in test_cases.iterrows()]

test_case1 = test_cases[0]