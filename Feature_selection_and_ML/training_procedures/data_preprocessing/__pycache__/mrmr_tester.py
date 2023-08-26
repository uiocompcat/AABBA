import pymrmr
from data_preprocessing import load_data
import pandas as pd
data_path = "/home/jeb/Desktop/ABBA_Paper/data/autocorrelation_vectors"
gp = "/ABBA_GP_d6.csv"
target_path = "/home/jeb/Desktop/ABBA_Paper/ac_generation/data_Vaska/data_27_april/"
target_gp = "gpVaska_vectors.csv"
target = "target_barrier" # Starting with barrier
df, target_vector = load_data(data_path + gp, target_path + target_gp, target)
df["target"] = target_vector
convert_dict = {}
for column in df.columns:
    convert_dict[column] = 'float64'

df = df.astype(convert_dict)
print(df.dtypes)
df = df.drop(columns="Unnamed: 0")
print(df.columns)
import mrmr
import polars
import pyspark
import pyspark.pandas as ps
# create some polars data
df_spark = ps.from_pandas(df)

# select top 2 features using mRMR
import mrmr
selected_features = mrmr.spark.mrmr_regression(df=df_spark, target_column="target", features=["Z-0_FA_AA", "Z-1_FA_AA", "Z-2_FA_AA"],K=2)
print(selected_features)
#df_polars = polars.DataFrame(data=df)
#selected_features = mrmr.polars.mrmr_regression(df=df_polars, target_column="target", K=2)

#selected_features = pymrmr.mRMR(df, "MIQ", 10)
#print(selected_features)
