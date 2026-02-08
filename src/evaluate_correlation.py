# qpp_corr_from_preds.py
import argparse
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau

def corr_from_file(path: str, other_metric: bool = False):
    df = pd.read_csv(path, sep="\t", dtype={"qid": str})
    # make sure columns exist
    if not {"pred", "label"}.issubset(df.columns):
        raise ValueError(f"{path} must contain columns: qid, pred, label")
    # numeric + drop rows without ground truth or prediction
    df["pred"]  = pd.to_numeric(df["pred"], errors="coerce")
    if other_metric:
        label_name = "label1"
    else:
        label_name = "label"
        
    df[label_name] = pd.to_numeric(df[label_name], errors="coerce")
    df = df.dropna(subset=["pred", label_name])

    if len(df) == 0:
        return {"n": 0, "pearson": None, "spearman": None, "kendall": None}

    p = df["pred"].to_numpy()
    y = df[label_name].to_numpy()
    pr = float(pearsonr(p, y)[0])
    sp = float(spearmanr(p, y).correlation)
    kt = float(kendalltau(p, y)[0])
    return {"n": int(len(df)), "pearson": pr, "kendall": kt, "spearman": sp}

def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("files", nargs="+", help="TSV files with columns: qid, pred, label")
    ap.add_argument("--save", help="Optional path to save a TSV summary")
    ap.add_argument("--input", help="Optional path to save a TSV summary")
    ap.add_argument("--collection", help="Optional path to save a TSV summary", default="V1")
    ap.add_argument("--other_metric_enable", action="store_true", help="Use other metric for correlation")

    args = ap.parse_args()

    rows = ''
    rows1 = ''
    if args.collection == 'V1':
        
        years = [ 'dev', '2019', '2020', 'hard']
        for year in years:
            res = corr_from_file(args.input + f'/pyg_graphs_{year}_pred.tsv', other_metric=False)
            print(f"{year}\t pearson  kendall  spearman : {res['pearson']}\t{res['kendall']}\t{res['spearman']}")
            rows +=f"{round(res['pearson'],3)}\t{round(res['kendall'],3)}\t{round(res['spearman'],3)}\t"
            if args.other_metric_enable:
                res = corr_from_file(args.input + f'/pyg_graphs_{year}_pred.tsv', other_metric=True)
                print(f"{year}\t pearson  kendall  spearman : {res['pearson']}\t{res['kendall']}\t{res['spearman']}")
                rows1 +=f"{round(res['pearson'],3)}\t{round(res['kendall'],3)}\t{round(res['spearman'],3)}\t"
                
        
    else:
        years = ['2021', '2022']
        for year in years:
            res = corr_from_file(args.input + f'/pyg_graphs_{year}_pred.tsv', other_metric=False)
            print(f"{year}\t pearson  kendall  spearman : {res['pearson']}\t{res['kendall']}\t{res['spearman']}")
            rows +=f"{round(res['pearson'],3)}\t{round(res['kendall'],3)}\t{round(res['spearman'],3)}\t"
            if args.other_metric_enable:
                res = corr_from_file(args.input + f'/pyg_graphs_{year}_pred.tsv', other_metric=True)
                print(f"{year}\t pearson  kendall  spearman : {res['pearson']}\t{res['kendall']}\t{res['spearman']}")
                rows1 +=f"{round(res['pearson'],3)}\t{round(res['kendall'],3)}\t{round(res['spearman'],3)}\t"
            
    if args.save:
        with open(args.save,'w') as f:
            f.write(rows)
            f.write('\n')
            f.write(rows1)
            
        # pd.DataFrame(rows).to_csv(args.save, sep="\t", index=False)

if __name__ == "__main__":
    main()