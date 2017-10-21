from datetime import datetime
from .logger import get_logger

logger = get_logger()

def write_submission_file(df, columns, name = 'model'):
    ts = datetime.now().strftime("%Y.%m.%d.%H.%M.%s")
    fname = "data/submissions/{}-{}.csv.gz".format(name, ts)
    df.to_csv(fname, columns = columns, compression = "gzip")
    logger.info("Created submission file {}".format(fname))
