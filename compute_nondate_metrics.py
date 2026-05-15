import polars as pl
import math
from coep.analysis.LLM_accuracy import make_scored_rows, make_dimension_rows

def main():
    input_path = "data/output/all_jurisdictions_benchmark_20260515_104348.csv"
    raw_df = pl.read_csv(input_path)
    
    # Get scored rows and dimensions
    scored_df = make_scored_rows(raw_df)
    dimension_df = make_dimension_rows(raw_df)
    
    # Filter out "Date" question type
    scored_df = scored_df.filter(pl.col("question_type") != "Date")
    dimension_df = dimension_df.filter(pl.col("question_type") != "Date")
    
    # Overall Metrics
    total_processed_queries = dimension_df["query_instance_id"].n_unique()
    
    def compute_weighted_score(df, n_queries):
        if n_queries == 0: return 0.0
        per_query = df.group_by("query_instance_id").agg([
            pl.len().alias("n_scored"),
            pl.col("eval_label").eq("Correct").sum().alias("n_correct")
        ])
        points_per_query = 100.0 / n_queries
        earned = (per_query.select((pl.lit(points_per_query) * pl.col("n_correct") / pl.col("n_scored")).sum())).item()
        return earned

    overall_weighted = compute_weighted_score(scored_df, total_processed_queries)
    
    # DPL / SSP
    dpl_scored = scored_df.filter(pl.col("dataset") == "DPL")
    dpl_dim = dimension_df.filter(pl.col("dataset") == "DPL")
    dpl_weighted = compute_weighted_score(dpl_scored, dpl_dim["query_instance_id"].n_unique())
    
    ssp_scored = scored_df.filter(pl.col("dataset") == "SSP")
    ssp_dim = dimension_df.filter(pl.col("dataset") == "SSP")
    ssp_weighted = compute_weighted_score(ssp_scored, ssp_dim["query_instance_id"].n_unique())

    print(f"Overall Non-Date Weighted Query Score: {overall_weighted:.2f}%")
    print(f"DPL Non-Date Weighted Query Score: {dpl_weighted:.2f}%")
    print(f"SSP Non-Date Weighted Query Score: {ssp_weighted:.2f}%")
    print("-" * 30)
    
    # Per Jurisdiction
    jurisdictions = dimension_df["jurisdiction"].unique().sort()
    print(f"{'Jurisdiction':<20} | {'Query Level Acc':<15} | {'Weighted Score':<15} | {'Count':<5}")
    for jur in jurisdictions:
        jur_scored = scored_df.filter(pl.col("jurisdiction") == jur)
        jur_dim = dimension_df.filter(pl.col("jurisdiction") == jur)
        n_queries = jur_dim["query_instance_id"].n_unique()
        
        if n_queries == 0:
            continue
            
        per_query = jur_scored.group_by("query_instance_id").agg([
            pl.len().alias("n_scored"),
            pl.col("eval_label").eq("Correct").sum().alias("n_correct"),
            pl.col("eval_label").eq("Correct").all().alias("all_correct")
        ])
        
        fully_correct_q = per_query.filter(pl.col("all_correct")).height
        q_acc = 100.0 * fully_correct_q / n_queries
        
        points_per_q = 100.0 / n_queries
        earned = (per_query.select((pl.lit(points_per_q) * pl.col("n_correct") / pl.col("n_scored")).sum())).item()
        
        print(f"{jur:<20} | {q_acc:>14.2f}% | {earned:>14.2f}% | {n_queries:<5}")

if __name__ == "__main__":
    main()
