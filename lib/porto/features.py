from lib.logger import get_logger

logger = get_logger()

def drop_cols(df, names = False):
    drop_cols = ["ps_calc_{:02d}".format(n) for n in range(2, 15)]
    drop_cols.append("ps_ind_12_bin")
    drop_idx = [i for i, name in enumerate(df.columns.values) if name in drop_cols]
    if names:
        return drop_cols
    else:
        return(drop_idx)
