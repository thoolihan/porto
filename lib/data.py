import boto3
from pathlib import Path
from .config import get_config
from .logger import get_logger
import pandas as pd

logger = get_logger()

def load_file(description = "train"):
    cfg = get_config()["data-files"][description]
    fname = cfg["file"].split("/")[-1]
    local_file = "data/{}".format(fname)

    if not(Path(local_file).exists()):
        s3 = boto3.resource('s3')
        try:
            logger.info("Saving S3 {}/{} to {}".format(cfg["bucket"], cfg["file"], local_file))
            s3.Bucket(cfg["bucket"]).Object(cfg["file"]).download_file(local_file)
            logger.info("Wrote {}".format(local_file))
        except Exception as e:
            logger.info("Exception getting file {}/{} from S3".format(cfg["bucket"], cfg["file"]))
    else:
        logger.info("Using already cached file {}".format(local_file))
    df = pd.read_csv(local_file, index_col = "id")
    if 'target' in df.columns:
        df.target  = df.target.astype('int')
    return df
