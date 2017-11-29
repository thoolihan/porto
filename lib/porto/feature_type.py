from lib.logger import get_logger

logger = get_logger()

def get_bin_cat_features(df):
    columns = list(df.columns.values)
    icols = [col for col in columns if col[-3:] in ['bin', 'cat']]
    logger.debug("Identified these columns to convert to int: {}".format(icols))
    return icols

def get_cat_features_idx(df):
    columns = list(df.columns.values)
    icols = [i for i, col in enumerate(columns) if col[-3:] == "cat"]
    logger.debug("Identified categorical columns: {}".format(icols))
    return icols

def get_cat_features(df):
    columns = list(df.columns.values)
    cols = [col for i, col in enumerate(columns) if col[-3:] == "cat"]
    logger.debug("Identified categorical columns: {}".format(cols))
    return cols