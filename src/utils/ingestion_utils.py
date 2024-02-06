import pandas as pd

def concat_x_y(X,y):
    #Concatenates X and y into a single DataFrame
    return pd.concat([X,pd.DataFrame(y, columns=['phishing'])],axis=1)


class DataCleaning:

    def __init__(self, df, missing_threshold, corr_threshold):
        self.df = df
        self.missing_threshold = missing_threshold
        self.corr_threshold = corr_threshold

    def drop_particular_features(self):
        skipped_features = [
            'time_response',
            'domain_spf',
            'asn_ip',
            'qty_ip_resolved',
            'qty_nameservers',
            'qty_mx_servers',
            'ttl_hostname',
            'qty_redirects',
            'url_google_index',
            'domain_google_index',
            'time_domain_activation',
            'time_domain_expiration'
        ]
        self.df = self.df.drop(skipped_features,axis=1)
        return self.df

    def col_with_variance_0(self):
        columns_to_drop = []
        numerical_columns = [col for col in self.df.columns if self.df[col].dtype != 'O']
        for col in numerical_columns:
            if self.df[col].std() == 0:
                columns_to_drop.append(col)
        return columns_to_drop

    def get_redundant_cols(self):
        cols_missing_ratios = self.df.isna().sum().div(self.df.shape[0])
        cols_to_drop = list(cols_missing_ratios[cols_missing_ratios > self.missing_threshold].index)
        return cols_to_drop

    def dropping_columns_on_basis_of_correlation(self):
        columns_to_drop = set()
        relation = self.df.corr()
        for columns in range(len(relation.columns)):
            for rows in range(columns):
                if abs(relation.iloc[columns, rows]) > self.corr_threshold:
                    col_name = relation.columns[columns]
                    columns_to_drop.add(col_name)
        columns_to_drop = list(columns_to_drop)
        return columns_to_drop

    def feature_scaling_df(self):
        self.df = self.drop_particular_features()
        cols_to_drop_1 = self.get_redundant_cols()
        cols_to_drop_2 = self.col_with_variance_0()
        cols_to_drop_3 = self.dropping_columns_on_basis_of_correlation()
        columns_to_drop = cols_to_drop_1 + cols_to_drop_2 + cols_to_drop_3
        columns_to_drop = set(columns_to_drop)
        self.df = self.df.drop(columns_to_drop,axis=1)
        self.df = self.df.drop_duplicates()
        return self.df
