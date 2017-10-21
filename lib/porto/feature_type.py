from lib.logger import get_logger

logger = get_logger()

def get_bin_cat_features(df):
    columns = list(df.columns.values)
    icols = [col for col in columns if col[-3:] in ['bin', 'cat']]
    logger.debug("Identified these columns to convert to int: {}".format(icols))
    return icols
