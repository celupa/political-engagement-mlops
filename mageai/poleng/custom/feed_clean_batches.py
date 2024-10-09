if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

from scenarios.feed_clean_batches import predict_prod_batches

@custom
def transform_custom(*args, **kwargs):
    predict_prod_batches()

