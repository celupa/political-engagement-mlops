if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

from scenarios.simulate import simulate_batches


@custom
def transform_custom(*args, **kwargs):
    simulate_batches()

