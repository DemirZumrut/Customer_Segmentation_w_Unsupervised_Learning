from src.utils import handsome_func, handsome_pca


def main():
    # handsome_pca function return pca-kmeans, famd-kmeans and pca-hierarchical cluster labels
    df_clustered = handsome_pca()
    df_rf_segment = handsome_func()
    # Add RFM_SEGMENT to df_clustered
    df_clustered["RFM_SEGMENT"] = df_rf_segment["RF_SEGMENT"]
    print(100 * "*")
    print(df_clustered.head())

    # General statistical analysis of each model's labels.
    group_list = [col for col in df_clustered.columns if
                  ("cluster" in col) or ("kmeans" in col) or ("RFM_SEGMENT" in col)]

    for group in group_list:
        print(100 * "*")
        print(30 * "*" + f" {group} " + 30 * "*")
        print(df_clustered.groupby(f"{group}").agg(["count", "mean", "median"]))


if __name__ == '__main__':
    main()
