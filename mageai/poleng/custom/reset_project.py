if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

from scripts.reset_project import reset_data
import os

@custom
def transform_custom(*args, **kwargs):
    os.system("python -m scripts.reset_project true")

