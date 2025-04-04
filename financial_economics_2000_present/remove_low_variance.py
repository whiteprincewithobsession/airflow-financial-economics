def remove_low_variance_cols(df, min_unique=5, max_cardinality_pct=1):
    unique_counts = {c: df.select(c).distinct().count() for c in df.columns}
    
    low_var_cols = [c for c, cnt in unique_counts.items() if cnt < min_unique]
    
    high_card_cols = []
    for c in [f.name for f in df.schema if f.dataType.typeName() in ["string"]]:
        unique_ratio = unique_counts[c] / df.count() * 100
        if unique_ratio > max_cardinality_pct:
            high_card_cols.append(c)
    
    cols_to_drop = list(set(low_var_cols + high_card_cols))
    return df.drop(*cols_to_drop)