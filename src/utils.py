import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import datetime as dt

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from light_famd import FAMD
from sklearn.cluster import AgglomerativeClustering

# display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.width', 500)

path_ = "/Users/ferhatkatar/PycharmProjects/VBO/Modul6_machine_learning/Homeworks/Company_X_Customer_Segmentation/Company_X_data_20k.csv"


def handsome_pca():
    ################################################
    # 1. Exploratory Data Analysis
    ################################################

    def load_data(path_):
        path = path_
        df = pd.read_csv(path)
        return df

    def check_df(dataframe, head=5):
        print("##################### Shape #####################")
        print(dataframe.shape)
        print("##################### Types #####################")
        print(dataframe.dtypes)
        print("##################### Head #####################")
        print(dataframe.head(head))
        print("##################### Tail #####################")
        print(dataframe.tail(head))
        print("##################### NA #####################")
        print(dataframe.isnull().sum())
        print("##################### Quantiles #####################")
        print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

    # The class frequencies and ratios of categorical variables.
    def cat_summary(dataframe, col_name, plot=False):
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")
        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

    # Numeric variables summary
    def num_summary(dataframe, numerical_col, plot=False):
        quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
        print(dataframe[numerical_col].describe(quantiles).T)

        if plot:
            dataframe[numerical_col].hist(bins=20)
            plt.xlabel(numerical_col)
            plt.title(numerical_col)
            plt.show(block=True)

    def correlation_matrix(df, cols):
        fig = plt.gcf()
        fig.set_size_inches(10, 8)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w',
                          cmap='RdBu')
        plt.show(block=True)

    def grab_col_names(dataframe, cat_th=10, car_th=32):
        # cat_cols, cat_but_car
        cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
        num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                       dataframe[col].dtypes != "O"]
        cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                       dataframe[col].dtypes == "O"]
        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]

        # num_cols
        num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
        num_cols = [col for col in num_cols if col not in num_but_cat]

        # print(f"Observations: {dataframe.shape[0]}")
        # print(f"Variables: {dataframe.shape[1]}")
        # print(f'cat_cols: {len(cat_cols)}')
        # print(f'num_cols: {len(num_cols)}')
        # print(f'cat_but_car: {len(cat_but_car)}')
        # print(f'num_but_cat: {len(num_but_cat)}')
        return cat_cols, num_cols, cat_but_car

    def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.99):
        quartile1 = dataframe[col_name].quantile(q1)
        quartile3 = dataframe[col_name].quantile(q3)
        interquantile_range = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * interquantile_range
        low_limit = quartile1 - 1.5 * interquantile_range
        return low_limit, up_limit

    def replace_with_thresholds(dataframe, variable):
        low_limit, up_limit = outlier_thresholds(dataframe, variable)
        dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
        dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

    def check_outlier(dataframe, col_name, q1=0.05, q3=0.99):
        low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
        if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
            return True
        else:
            return False

    # df.describe(cumstomized_perc, include='all').T
    # For numeric data, the result's index will include ``count``,
    #         ``mean``, ``std``, ``min``, ``max`` as well as lower, ``50`` and
    #         upper percentiles. By default the lower percentile is ``25`` and the
    #         upper percentile is ``75``. The ``50`` percentile is the
    #         same as the median.
    # cumstomized_perc = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    # df.describe(cumstomized_perc).T

    ################################################
    # 2. Data Preprocessing & Feature Engineering
    ################################################

    # df.describe(cumstomized_perc).T

    #########################
    # skewness and kurtosis
    #########################
    # Compute the sample skewness of a data set.
    #
    # For normally distributed data, the skewness should be about zero. For unimodal continuous distributions,
    # a skewness value greater than zero means that there is more weight in the right tail of the distribution.
    # The function skewtest can be used to determine if the skewness value is close enough to zero, statistically speaking.

    # for col in df[num_cols].columns:
    #     print(15 * "*" + "  " + col + "  " + 15 * "*")
    #     print("Skewness for the column {} is {}".format(col, df[col].skew()))
    #     print("kurtosis for the column {} is {}".format(col, df[col].kurtosis()))

    # ***************  order_num_total_ever_online  ***************
    # Skewness for the column order_num_total_ever_online is 4.52395978182151
    # kurtosis for the column order_num_total_ever_online is 32.451101691506786
    # ***************  order_num_total_ever_offline  ***************
    # Skewness for the column order_num_total_ever_offline is 3.2979449375938388
    # kurtosis for the column order_num_total_ever_offline is 18.733085466004507
    # ***************  customer_value_total_ever_offline  ***************
    # Skewness for the column customer_value_total_ever_offline is 3.633123066816902
    # kurtosis for the column customer_value_total_ever_offline is 24.052339855974452
    # ***************  customer_value_total_ever_online  ***************
    # Skewness for the column customer_value_total_ever_online is 4.317768901107312
    # kurtosis for the column customer_value_total_ever_online is 29.517863656000777

    def outlier_smoothing(df):
        cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=13, car_th=32)

        # check_outlier
        print(100 * "*")
        print("BEFORE OUTLIER SMOOTHING")
        print(100 * "*")

        for col in num_cols:
            print(col, check_outlier(df, col))

        # replace_with_thresholds
        for col in num_cols:
            outlier_thresholds(df, col)
            replace_with_thresholds(df, col)

        # check_outlier
        print(100 * "*")
        print("AFTER OUTLIER SMOOTHING")
        print(100 * "*")

        for col in num_cols:
            print(col, check_outlier(df, col))

        return df

    def feature_engineering(df):
        #################################
        # feature_engineering Steps:
        #################################

        # 1)  Calculate Total order for omnichannel {online + offline}
        # 2)  Calculate Total cost for omnichannel {online + offline}
        # 3)  Convert date variables from object to datetime format
        # 4)  Calculate for recency
        # 5)  Apply qcut for recency, frequency and monetary_score
        # 6)  Extract Insight from interested_in_categories_12 variable
        # 7)  Apply partition_date function for date variables.
        # 8)  Calculate some ratio values on customer values & order numbers.
        # 9)  Check channel homogeneity.
        # 10) Check channel Is_App
        # 11) one_hot_encoder
        # 12) RobustScaler
        # 13) Correlation Analyzing

        print(100 * "*")
        print("feature_engineering")
        print(100 * "*")

        # 1)  Calculate Total order for omnichannel {online + offline}
        df['total_num_order'] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

        # 2)  Calculate Total cost for omnichannel {online + offline}
        df["total_purchase"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

        # 3)  Convert date variables from object to datetime format
        obj_to_datetime = [col for col in df.columns if "date" in col]
        # Converting  from object to datetime format
        df[obj_to_datetime] = df[obj_to_datetime].apply(pd.to_datetime)

        """
        # Option 2
        # df.iloc[:,df.columns.str.contains("date")] = df.iloc[:,df.columns.str.contains("date")].astype('datetime64')
        """

        # 4)  Calculate for recency

        # Analyzing date
        last_tmsp = df["last_order_date"].max()
        # Timestamp('2021-05-30 00:00:00')

        # Recency Date
        analysing_date = dt.datetime(2021, 6, 1)

        """
        Option 2:
        today_date = df["last_order_date"].max() + dt.timedelta(days=2)
        """

        df["recency"] = df["last_order_date"].apply(lambda x: (analysing_date - x).days)

        # 5)  Apply qcut to create recency score, frequency score and monetary_score.

        df["recency_score"] = pd.qcut(df['recency'], 5, labels=[5, 4, 3, 2, 1])

        df["frequency_score"] = pd.qcut(df['total_num_order'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

        df["monetary_score"] = pd.qcut(df['total_purchase'], 5, labels=[1, 2, 3, 4, 5])

        df_final = df

        df_final.head()

        # 6)  Extract Insight from interested_in_categories_12 variable

        categ_list = list(
            set(''.join(df.interested_in_categories_12.unique()).replace(']', ',').replace('[', ',').replace(',,',
                                                                                                             ',').replace(
                ' ', '').split(',')))

        categ_list = categ_list[1:]

        for cat in categ_list:
            for ind, cat12 in enumerate(df.interested_in_categories_12):
                if cat in cat12:
                    df_final.loc[ind, cat] = 1
                else:
                    df_final.loc[ind, cat] = 0

        # 7)  Apply partition_date function for date variables.
        def partition_date(col_name, dataframe):
            dataframe[col_name + 'year'] = dataframe[col_name].dt.year
            dataframe[col_name + 'month'] = dataframe[col_name].dt.month
            dataframe[col_name + 'day'] = dataframe[col_name].dt.day

            dataframe[col_name + 'dayofweek_num'] = dataframe[col_name].dt.dayofweek
            dataframe[col_name + 'Week_Number'] = dataframe[col_name].dt.week
            dataframe[col_name + "Is Weekend"] = dataframe[col_name].dt.dayofweek > 4
            return dataframe

        for col in obj_to_datetime:
            df_final = partition_date(col, df_final)

        # 8)  Calculate some ratio values on customer values & order numbers.

        df_final["order_online_ratio"] = df_final["order_num_total_ever_online"] / df_final["total_num_order"]
        df_final["customer_value_online_ratio"] = df_final["customer_value_total_ever_online"] / df_final[
            "total_purchase"]
        df_final["purchase_per_order"] = df_final["total_purchase"] / df_final["total_num_order"]

        # Step 6 continue...

        df_final["IS_FAMILY_W_KID"] = 0

        df_final.loc[
            (((df_final["ERKEK"] == 1) | (df_final["KADIN"] == 1)) & (
                    (df_final["COCUK"] == 1) | (df_final["AKTIFCOCUK"] == 1))),
            "IS_FAMILY_W_KID"] = 1

        df_final["IS_FAMILY_NO_KID"] = 0

        df_final.loc[
            (((df_final["ERKEK"] == 1) & (df_final["KADIN"] == 1)) & (
                    (df_final["COCUK"] == 0) & (df_final["AKTIFCOCUK"] == 0))),
            "IS_FAMILY_NO_KID"] = 1

        df_final["IS_Sportive"] = 0

        df_final.loc[(df_final["AKTIFCOCUK"] == 1) | (df_final["AKTIFSPOR"] == 1), "IS_Sportive"] = 1

        # 9)  Check channel homogeneity.

        df_final["Is_homojenity"] = 0

        df_final.loc[(df_final["order_channel"] == df_final["last_order_channel"]), "Is_homojenity"] = 1

        # 10) Check channel Is_App

        # APP -- Using vectorization method for efficient flow.
        df_final["Is_App"] = 0
        m = df_final.order_channel.str.contains('App') | df_final.last_order_channel.str.contains('App')
        df_final['Is_App'] = np.where(m, 1, df_final["Is_App"])

        # Option 2:
        # from pandasql import sqldf
        # q = """SELECT CASE
        #                 WHEN ((order_channel like '%App%')OR( CONTAINS(last_order_channel, 'App')) then 1
        #                 else 0 end is_app
        #        FROM df order_channel"""
        # df['is_app'] = sqldf(q, globals())

        ##############

        cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=13, car_th=32)

        # 11) one_hot_encoder
        def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
            dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
            return dataframe

        df_scale = df_final

        # df_scale = one_hot_encoder(df, cat_cols, drop_first=True)

        cat_cols, num_cols, cat_but_car = grab_col_names(df_scale, cat_th=13, car_th=32)

        columns_to_drop = ['first_order_date',
                           'last_order_date',
                           'last_order_date_online',
                           'last_order_date_offline',
                           'master_id']

        scale_cols = [col for col in num_cols if col not in columns_to_drop]

        df_scale[scale_cols].head()

        # 12) RobustScaler
        X_scaled = RobustScaler().fit_transform(df_scale[scale_cols])
        df_scale[scale_cols] = pd.DataFrame(X_scaled, columns=df_scale[scale_cols].columns)

        """
        plt.figure(figsize=(50, 50))
        hm = sns.heatmap(df_scale.corr(), annot=True)
        hm.set(xlabel='\nFLO Variables Details', ylabel='FLO Variables Details\t',
               title="Correlation matrix of FLO data\n")
        plt.show()
        """
        #######################
        # 13) Correlation Analyzing
        # The main goal of this Correlation Analyzing is to break the multicollinearity between variables.
        # Drop correlated columns which has more total correlation with other columns

        cor_matrix = df_scale.corr().abs()
        cor_matrix

        cor_matrix.where((np.triu(np.ones(cor_matrix.shape), k=1) +
                          np.tril(np.ones(cor_matrix.shape), k=-1)).astype(bool))

        cor_matrix["corr_sum"] = 0.0

        for col in range(cor_matrix.shape[1] - 1):
            cor_matrix["corr_sum"][col] = cor_matrix.iloc[:, col:col + 1].sum()

        cor_matrix["corr_sum"]
        upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
        upper_tri

        # Drop correlated columns which has more total correlation with other columns.
        to_drop_list = []
        for k in range(upper_tri.shape[1] - 1):
            for i in range(upper_tri.shape[1] - 1):
                if (upper_tri.iloc[k][i] > 0.9) == True:
                    if upper_tri.iloc[k][upper_tri.shape[1] - 1] > upper_tri.iloc[i][upper_tri.shape[1] - 1]:
                        to_drop_list.append(upper_tri.index[k])
                    else:
                        to_drop_list.append(upper_tri.index[k])
                    print(upper_tri.iloc[k][i], k, i)

        to_drop_list = np.unique(to_drop_list)
        to_drop_list

        df_scale.drop(columns=to_drop_list, axis=1, inplace=True)

        df_scale.head()

        df_scale.head()
        df_scale.drop(columns=columns_to_drop, axis=1, inplace=True)
        return df_scale

    def apply_pca(df_scale):

        print(100 * "*")
        print("apply_pca")
        print(100 * "*")
        cat_cols, num_cols, cat_but_car = grab_col_names(df_scale, cat_th=15, car_th=33)

        pca = PCA().fit(df_scale[num_cols])

        # cumulative variance(information) ratio
        pca.explained_variance_ratio_
        print("cumulative variance")
        print(np.cumsum(pca.explained_variance_ratio_))

        # ALL
        # [0.22452679 0.43412512 0.58513668 0.68766391 0.7832598  0.86366542
        #  0.92604252 0.95290002 0.97360635 0.99033588 0.99597997 1.        ]

        def create_pca_df(X, n_components=6):
            pca = PCA(n_components=n_components)
            pca_fit = pca.fit_transform(X)
            columns = []
            for i in range(1, n_components + 1):
                i = str(i)
                columns.append("PC" + i)
            pca_df = pd.DataFrame(data=pca_fit, columns=columns)
            final_df = pca_df
            return final_df

        pca_df = create_pca_df(df_scale[num_cols], n_components=6)

        return pca_df

    def apply_famd(df_scale):

        """
        This article presents the Factorial Analysis of Mixed Data (FAMD), which generalizes the
        Principal Component Analysis (PCA) algorithm to datasets containing numerical and categorical variables.

        Details Resource:
        https://towardsdatascience.com/famd-how-to-generalize-pca-to-categorical-and-numerical-data-2ddbeb2b9210
        """

        print(100 * "*")
        print("apply_famd")
        print(100 * "*")

        obj_bool = [col for col in df_scale.columns if df_scale[col].dtypes in ["category", "bool"]]
        df_scale[obj_bool] = df_scale[obj_bool].astype("object")

        # [pd.api.types.is_string_dtype(df_scale[c]) for c in obj_bool]

        famd = FAMD(n_components=30)
        famd.fit_transform(df_scale)
        print(np.cumsum(famd.explained_variance_ratio_))

        # [0.30716126 0.37431787 0.42689184 0.46681893 0.49671635 0.52184679
        #  0.54411331 0.56428457 0.5839998  0.60323974 0.62195681 0.64011591
        #  0.657961   0.67481796 0.69108447 0.70588435 0.72044923 0.73474599
        #  0.74888327 0.76288298 0.77679896 0.79035763 0.80274856 0.81492628
        #  0.82656938 0.83805326 0.84942368 0.86063535 0.87118193 0.87948628]

        def create_famd_df(X, n_components=30):
            famd1 = FAMD(n_components=n_components)
            famd_fit = famd1.fit_transform(X)
            columns = []
            for i in range(1, n_components + 1):
                i = str(i)
                columns.append("FAMD" + i)
            famd_df = pd.DataFrame(data=famd_fit, columns=columns)
            final_df = famd_df
            return final_df

        famd_df = create_famd_df(df_scale, n_components=30)
        return famd_df

    def kmeans_model(dataframe, source):

        # Source parameter defines which dataframe source will be using for kmeans model.
        # There are three options as follows --> Original Scaled Dataframe - PCA Dataframe - FAMD Dataframe

        ########################
        # KMEANS
        #########################

        print(100 * "*")
        print("KMEANS Model")
        print(100 * "*")
        print(f"Kmeans is applying on {source} for now")
        print(100 * "*")

        if source == "Original Scaled Dataframe":
            cat_cols, num_cols, cat_but_car = grab_col_names(df_scale, cat_th=15, car_th=33)
            dataframe = dataframe[num_cols]

        kmeans = KMeans()
        ssd = []
        K = range(1, 30)

        for k in K:
            kmeans = KMeans(n_clusters=k).fit(dataframe)
            ssd.append(kmeans.inertia_)

        # Optimum cluster number by using elbow:
        kmeans = KMeans()
        elbow = KElbowVisualizer(kmeans, k=(2, 20))
        elbow.fit(dataframe)
        # elbow.show()

        kmeans = KMeans(n_clusters=elbow.elbow_value_, random_state=17).fit(dataframe)
        kmeans.get_params()

        # Optimum cluster number
        kmeans.n_clusters

        # Centers of 8 Clusters Determined as Final:
        kmeans.cluster_centers_

        # contains the class label for each observation.
        kmeans.labels_

        kmeans.inertia_
        #     inertia_ : float
        #         Sum of squared distances(SSD-->sum of squared distances) of samples to their closest cluster center,
        #         weighted by the sample weights if provided.
        clusters_kmeans_labels = kmeans.labels_
        return clusters_kmeans_labels

    def hierarchical_clustering(dataframe, source, plot=False):

        # Source parameter defines which dataframe source will be using for HIERARCHICAL CLUSTERING.
        # There are three options as follows --> Original Scaled Dataframe - PCA Dataframe - FAMD Dataframe

        ########################
        # HIERARCHICAL CLUSTERING
        #########################

        print(100 * "*")
        print("HIERARCHICAL CLUSTERING")
        print(100 * "*")
        print(f"HIERARCHICAL CLUSTERING is applying on {source} for now")
        print(100 * "*")

        if source == "Original Scaled Dataframe":
            cat_cols, num_cols, cat_but_car = grab_col_names(df_scale, cat_th=15, car_th=33)
            dataframe = dataframe[num_cols]

        hc_average = linkage(dataframe, "average")

        if plot:
            plt.figure(figsize=(16, 8))
            plt.title("hierarchical Cluster Dendogram")
            plt.xlabel("Rows")
            plt.ylabel("Distances")
            dendrogram(hc_average,
                       truncate_mode="lastp",
                       p=20,
                       show_contracted=True,
                       leaf_font_size=10,
                       show_leaf_counts=True)
            plt.axhline(y=10, color='r', linestyle='--')
            plt.axhline(y=7, color='b', linestyle='--')
            plt.show()

        cluster = AgglomerativeClustering(n_clusters=8, linkage="average")

        hierarchical_clustering_labels = cluster.fit_predict(dataframe)
        return hierarchical_clustering_labels

    def add_label_dataframe(df_original, clusters_kmeans_for_original_data, clusters_kmeans_for_pca,
                            clusters_kmeans_for_famd, hi_cluster_for_original_data, hi_cluster_for_pca,
                            hi_cluster_for_famd):

        # The cluster labels of each model results will be added to the original dataframe.

        print(100 * "*")
        print("add_label_dataframe")
        print(100 * "*")

        # kmeans
        df_original["kmeans_for_original_data"] = clusters_kmeans_for_original_data
        df_original["kmeans_for_original_data"] = df_original["kmeans_for_original_data"] + 1

        df_original["kmeans_for_pca"] = clusters_kmeans_for_pca
        df_original["kmeans_for_pca"] = df_original["kmeans_for_pca"] + 1

        df_original["kmeans_for_famd"] = clusters_kmeans_for_famd
        df_original["kmeans_for_famd"] = df_original["kmeans_for_famd"] + 1

        # hierarchical_clustering
        df_original["hi_cluster_for_original_data"] = hi_cluster_for_original_data
        df_original["hi_cluster_for_original_data"] = df_original["hi_cluster_for_original_data"] + 1

        df_original["hi_cluster_for_pca"] = hi_cluster_for_pca
        df_original["hi_cluster_for_pca"] = df_original["hi_cluster_for_pca"] + 1

        df_original["hi_cluster_for_famd"] = hi_cluster_for_famd
        df_original["hi_cluster_for_famd"] = df_original["hi_cluster_for_famd"] + 1

        return df_original

    df = load_data(path_)
    df_original = df.copy()

    df_smoothed = outlier_smoothing(df)

    df_scale = feature_engineering(df_smoothed)

    pca_df = apply_pca(df_scale)
    famd_df = apply_famd(df_scale)

    # There are three options as follows --> Original Scaled Dataframe - PCA Dataframe - FAMD Dataframe

    # KMEANS for original scaled dataframe, pca dataframe, famd dataframe respectively.
    # kmeans_model function is returned related model's cluster labels.
    clusters_kmeans_for_original_data = kmeans_model(df_scale, source="Original Scaled Dataframe")
    clusters_kmeans_for_pca = kmeans_model(pca_df, source="PCA Dataframe")
    clusters_kmeans_for_famd = kmeans_model(famd_df, source="FAMD Dataframe")

    # hierarchical for original scaled dataframe, pca dataframe, famd dataframe respectively.
    # hierarchical_clustering function is returned related model's cluster labels.
    hi_cluster_for_original_data = hierarchical_clustering(df_scale, source="Original Scaled Dataframe")
    hi_cluster_for_pca = hierarchical_clustering(pca_df, source="PCA Dataframe")
    hi_cluster_for_famd = hierarchical_clustering(famd_df, source="FAMD Dataframe")

    df_labelled = add_label_dataframe(df_original, clusters_kmeans_for_original_data, clusters_kmeans_for_pca,
                                      clusters_kmeans_for_famd, hi_cluster_for_original_data, hi_cluster_for_pca,
                                      hi_cluster_for_famd)

    return df_labelled


def handsome_func():
    # author : ARDA BAYSALLAR
    # DATE : 5 July 2022
    # RESOURCE : MIUUL - VBO Data Science Bootcamp 2022

    # -----------------------------------------------------------------------------------------------------
    # IMPORTS
    # -----------------------------------------------------------------------------------------------------

    print(100 * "*")
    print("handsome_func")
    print(100 * "*")

    import datetime

    # *****************************************************************************************************
    # -----------------------------------------------------------------------------------------------------
    # MISSION 1 : Understanding data and Data Preperation
    # -----------------------------------------------------------------------------------------------------
    # *****************************************************************************************************

    # Step 1 : Read data and copy for future requirements
    # -----------------------------------------------------------------------------------------------------

    df_0 = pd.read_csv(path_)
    df = df_0.copy()

    # Step 2 : Data Observation and Stats
    # -----------------------------------------------------------------------------------------------------

    # NOTES :  order_num_total_ever_online/offline seems to have outliers because mean shows 3.11/1.91 orders
    # with very moderate std and 75% quantile however, the maximum number of orders is 200 / 109
    # the same situation is also showing itself for customer values
    # -----------------------------------------------------------------------------------------------------

    # missing values :
    df.isnull().sum()  # there is no missing values present itself in this analysis

    # number of uniques for categorical and numericals
    def basic_cat_num_analyzer(df):
        cols = df.columns
        cat_cols = [col for col in cols if str(df[col].dtype) in ('object', 'category')]
        num_cols = [col for col in cols if col not in cat_cols]
        cats_nuniq = pd.DataFrame({col: df[col].nunique() for col in cat_cols}, index=[0])
        nums_nuniq = pd.DataFrame({col: df[col].nunique() for col in num_cols}, index=[0])

        return cat_cols, num_cols, cats_nuniq, nums_nuniq

    cats_unique, nums_unique = basic_cat_num_analyzer(df)[2:4]

    # Step 3 : Omnichannel : customers shop from both online and offline platforms.
    # Create new variables for each customer's total shopping count and spending.
    # -----------------------------------------------------------------------------------------------------

    df['order_num_total_ever_omni'] = df['order_num_total_ever_offline'] + df['order_num_total_ever_online']
    df['customer_value_total_ever_omni'] = df['customer_value_total_ever_offline'] + df[
        'customer_value_total_ever_online']

    # Step 4 : Feature types , date type transformation
    # -----------------------------------------------------------------------------------------------------
    # date columns detection from column name :
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    df[date_cols] = df[date_cols].astype('datetime64[ns]')

    # Step 5 : See the distribution of the number of customers in shopping channels,
    # the number of products collected and their total expenditures.
    # -----------------------------------------------------------------------------------------------------
    df_agg_channel = df.groupby('order_channel').agg({'order_num_total_ever_omni': ['sum'],
                                                      'customer_value_total_ever_omni': ['sum']
                                                      })
    df_agg_channel.columns = df_agg_channel.columns.droplevel(1)
    df_agg_channel.reset_index(inplace=True)
    df_agg_channel['channel_order_percentage'] = 100 * (df_agg_channel.order_num_total_ever_omni) / (
        df_agg_channel.order_num_total_ever_omni.sum())
    df_agg_channel['channel_value_percentage'] = 100 * (df_agg_channel.customer_value_total_ever_omni) / (
        df_agg_channel.customer_value_total_ever_omni.sum())
    df_agg_channel.sort_values(by=df_agg_channel.columns.tolist(), ascending=False, inplace=True)

    axes = df_agg_channel.set_index('order_channel').iloc[:, 2:].plot.bar(rot=0, subplots=True)
    # NOTES :  it is showing that Android App is most used channel and the other channels are showing same
    # distribution for both volume and count
    # -----------------------------------------------------------------------------------------------------

    # Step 6 : Top 10 customer with highest revenue value
    # -----------------------------------------------------------------------------------------------------
    df.sort_values(by='customer_value_total_ever_omni', ascending=False).head(10)

    df.sort_values(by='customer_value_total_ever_omni', ascending=False).head(10).describe().T

    # Step 7 : Top 10 customer with highest number of orders
    # -----------------------------------------------------------------------------------------------------
    df.sort_values(by='order_num_total_ever_omni', ascending=False).head(10)
    df.sort_values(by='order_num_total_ever_omni', ascending=False).head(10).describe().T

    # Step 8 : FUNCTIONAL DATA PREPERATION
    # -----------------------------------------------------------------------------------------------------
    # STEP 3 : OMNI CHANNEL COLUMNS
    def omni_col_creator(df):
        """
        Create omni-channel count and volume columns for dataframe
        Parameters
        ----------
        df : pandas dataframe

        Returns
        -------
        new data frame with old dataframe + columns: order_num_total_ever_omni & customer_value_total_ever_omni
        same dataframe with added new 2 columns one for omni-channel offline and online total counts
        the other one offline and online total purchase volume per customer

        df : pandas dataframe

        """
        df['order_num_total_ever_omni'] = df['order_num_total_ever_offline'] + df['order_num_total_ever_online']
        df['customer_value_total_ever_omni'] = df['customer_value_total_ever_offline'] + df[
            'customer_value_total_ever_online']
        return df

    # STEP 4 : DTYPE CONVERSION
    def date_column_finder(df):
        """
        Find the columns that has date in their name
        Parameters
        ----------
        df : pandas dataframe

        Returns
        -------
        date_columns_list : list
        """
        date_cols = [col for col in df.columns if 'date' in col.lower()]

        return date_cols

    def date_convertor(df):
        """
        find date columns and covert dtype to date type
        Parameters
        ----------
        df : pandas dataframe

        Returns
        -------

        new dataframe that transforms old dataframe's date columns to dtype = datetime64[ns]
        and detected date columns as a list

        dataframe : pandas dataframe
        date_columns : list
        """

        # date columns detection from column name :
        date_cols = date_column_finder(df)
        df[date_cols] = df[date_cols].astype('datetime64[ns]')

        return df, date_cols

    def chan_agg(df):
        """
        Channel based aggregation for omni channel sales and volume
        Parameters
        ----------
        df : pandas dataframe

        Returns
        -------
        Aggregated dataframe for channel groupby
        it has information of :
            * order_num_total_ever_omni : total count per channel
            * customer_value_total_ever_omni : total volume per channel
            * channel_order_percentage : total count percentage per channel
            * channel_value_percentage : total volume percentage per channel

        df_agg_channel : pandas dataframe

        """
        df_agg_channel = df.groupby('order_channel').agg({'order_num_total_ever_omni': ['sum'],
                                                          'customer_value_total_ever_omni': ['sum']
                                                          })

        df_agg_channel.columns = df_agg_channel.columns.droplevel(1)  # drop multiindex level 1 'sum'
        df_agg_channel.reset_index(inplace=True)  # reset indexes to iterative 0-len(df)

        # percentages
        df_agg_channel['channel_order_percentage'] = 100 * (df_agg_channel.order_num_total_ever_omni) / \
                                                     (df_agg_channel.order_num_total_ever_omni.sum())
        df_agg_channel['channel_value_percentage'] = 100 * (df_agg_channel.customer_value_total_ever_omni) / \
                                                     (df_agg_channel.customer_value_total_ever_omni.sum())
        # sorting descending
        df_agg_channel.sort_values(by=df_agg_channel.columns.tolist(), ascending=False, inplace=True)

        return df_agg_channel

    # All together MAIN
    def data_prep(df, channel_analysis=False):
        """
        Main function do the data preperation steps all in one func
        Parameters
        ----------
        df : pandas dataframe

        Returns
        -------
        return altered dataframe df_altered and date columns list

        df_altered : pandas dataframe
        date_columns : list

        if channel_analysis is True
        then it will give aggregated channel distribution analysis dataframe by volume and count
        else it will return emtpy dataframe
        """

        # omni channel columns
        df_omni = omni_col_creator(df)
        # date columns converter
        df_prep, date_columns = date_convertor(df_omni)

        if channel_analysis:
            df_agg_chan = chan_agg(df_prep)
        else:
            df_agg_chan = pd.DataFrame()

        return df_prep, date_columns, df_agg_chan

    df = df_0.copy()
    df, date_columns, df_agg_chan = data_prep(df)

    # *****************************************************************************************************
    # -----------------------------------------------------------------------------------------------------
    # MISSION 2 : RFM Metric Calculation
    # -----------------------------------------------------------------------------------------------------
    # *****************************************************************************************************

    # STEP 1 AND 2 : RECENCY - FREQUENCY - MONETARY

    # RECENCY : The last time customer buy something
    # the last time customer buys something is the last_order_date
    # analyze_date : date of analysis made after the last date of the latest purchase

    def recency_calc(df, give_buffer_days=1):
        """
        Create RECENCY column for RFM SEGMENTATION
        Parameters
        ----------
        df : pandas dataframe
        give_buffer_days : buffer date for analyze, default next day after latest transaction

        Returns
        -------
        df : pandas dataframe with RECENCY additional column with number of days integer
        """
        # ANALYZE DATE
        analyze_date = max(df.last_order_date) + datetime.timedelta(days=give_buffer_days)
        df['RECENCY'] = (analyze_date - df['last_order_date']).dt.days  # GIVES INTEGER DAYS

        return df

    # FREQUENCY  : Number of transaction per customer
    # MONETARY : Volume of total transactions per customers

    def rename_freq_and_monetary(df):
        """
        Set name of the order_num_total_ever_omni as FREQUENCY and
          customer_value_total_ever_omni as MONETARY column for RFM SEGMENTATION
        Parameters
        ----------
        df : pandas data frame

        Returns
        -------
        df : pandas dataframe with FREQUENCY AND MONETARY

        """

        df = df.rename(columns={'order_num_total_ever_omni': 'FREQUENCY',
                                'customer_value_total_ever_omni': 'MONETARY'})

        return df

    def rfm_col_generator(df, give_buffer_days=1):
        """
        Main RFM column generator
        Parameters
        ----------
        df : pandas dataframe
        give_buffer_days = 1 default

        Returns
        -------
        dataframe with RFM columns
        """
        df_m = rename_freq_and_monetary(df)
        df_rfm = recency_calc(df_m, give_buffer_days=give_buffer_days)

        df_rfm = df_rfm.round({'MONETARY': 0, 'FREQUENCY': 0})
        # STEP 3 : RFM COLUMN
        df_rfm['RFM'] = (df_rfm['RECENCY'].astype(str)
                         + df_rfm['FREQUENCY'].astype(str)
                         + df_rfm['MONETARY'].astype(str))

        return df_rfm

    df = rfm_col_generator(df, 1)

    # *****************************************************************************************************
    # -----------------------------------------------------------------------------------------------------
    # MISSION 3 : RF SCORE CALCULATION
    # -----------------------------------------------------------------------------------------------------
    # *****************************************************************************************************

    # SINCE FREQUENCY LARGER --> BETTER it will go from 1 to 5 because qcut sorts ascending
    # SINCE RECENCY SMALLER --> BETTER it will go from 5 to 1 because qcut sorts  ascending

    # STEP 1 & 2  : QCUT

    df['RECENCY_SCORE'] = pd.qcut(df['RECENCY'], 5, labels=[5, 4, 3, 2, 1])
    df['FREQUENCY_SCORE'] = pd.qcut(df['FREQUENCY'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    df['MONETARY_SCORE'] = pd.qcut(df['MONETARY'], 5, labels=[1, 2, 3, 4, 5])

    # STEP 3
    df['RF_SCORE'] = (df['RECENCY_SCORE'].astype(str) +
                      + df['FREQUENCY_SCORE'].astype(str))

    # *****************************************************************************************************
    # -----------------------------------------------------------------------------------------------------
    # MISSION 4 : RF SEGMENTATION
    # -----------------------------------------------------------------------------------------------------
    # *****************************************************************************************************
    # RFM REGEX MAP :
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_Risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'}

    def rf_segment_mapping(dataframe, seg_map={r'[1-2][1-2]': 'hibernating',
                                               r'[1-2][3-4]': 'at_Risk',
                                               r'[1-2]5': 'cant_loose',
                                               r'3[1-2]': 'about_to_sleep',
                                               r'33': 'need_attention',
                                               r'[3-4][4-5]': 'loyal_customers',
                                               r'41': 'promising',
                                               r'51': 'new_customers',
                                               r'[4-5][2-3]': 'potential_loyalists',
                                               r'5[4-5]': 'champions'}):
        """
        Returns the corresponding mapped segment according to Recency and Frequency Scores from (1-5) for both

        Parameters
        ----------
        seg_map : Dictionary -> default segment mapping is given
        dataframe : pandas dataframe

        Returns
        given dataframe with added SEGMENT column according to Recency and Frequency mapping explained below
        -------
        dataframe : Pandas Dataframe with column SEGMENT added according to rules :

            SEGMENT column with respect to the mapped RF score segments as follows :
            R->[1-2] + F->[1-2] : hibernating
            R->[1-2] + F->[3-4] : at-Risk
            R->[1-2] + F->5     : cant_loose
            R->3     + F->[1-2] : about_to_sleep
            R->3     + F-3      : need_attention
            R->[3-4] + F->[4-5] : loyal_customers
            R->4     + F->1     : promising
            R->5     + F->1     : new_customers
            R->[4-5] + F->[2-3] : potential_loyalists
            R->5     + F->[4-5] : champions

        """
        dataframe['RF_SEGMENT'] = dataframe['RF_SCORE'].replace(seg_map, regex=True)
        return dataframe

    df = rf_segment_mapping(df)

    return df
