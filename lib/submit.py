from datetime import datetime
import gzip
import shutil

def write_submission_file(df, columns, name = 'model'):
    ts = datetime.now().strftime("%Y.%m.%d.%H.%M.%s")
    fname = "data/submission-{}-{}.csv".format(name, t)
    df.to_csv(fname, columns = columns)
    with open(fname, 'rb') as f_in:
        with gzip.open("{}.gz".format(fname), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            
    
    
