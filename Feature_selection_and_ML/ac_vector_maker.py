import pandas as pd
import os

walks = ["AA", "BB", "BBavg", "AB"]
ACs = {"AA": ["FA", "FD", "FR", "FS", "MA", "MD", "MR", "MS"],
       "BB": ["FA", "FD", "FR", "FS", "MA", "MD", "MR", "MS"],
       "BBavg": ["MA", "MD", "MR", "MS"],
       "AB": ["FA", "FD", "FR", "FS", "MA", "MD", "MR", "MS"]
       }

"""
Periodic property set (PT)
d - depth
"""
prefix = "PT"
d = 6
path = os.getcwd()
path += "/data/Vaskas_project/"
df = pd.read_csv(path + f"PT_AA/FA_AA_d{str(d)}.csv")
for walk in walks:
    for AC in ACs[walk]:
        df2 = pd.read_csv(path + f"{prefix}_{walk}/{AC}_{walk}_d{str(d)}.csv")
        print(f"Shape {AC}_{walk} = {df2.shape}")
        df = pd.concat([df, df2.reindex(df.index)], axis=1)
print("Before dropping duplicates shape: ", df.shape)
PT_df = df.groupby(level=0).first().T
PT_df= df.T.groupby(level=0).first().T
print("After dropping duplicates shape: ", PT_df.shape)
print(PT_df)
print(f"Shape PT: {PT_df.shape}")
PT_df.to_csv(os.getcwd() + f"/data/autocorrelation_vectors/ABBA_GP_d{str(d)}.csv")

"""
Electronic property set (NBO)
"""
prefix = "NBO"
path = os.getcwd()
path += "/data/Vaskas_project/"
df = pd.read_csv(path + f"NBO_AA/FA_AA_d{str(d)}.csv")
for walk in walks:
    for AC in ACs[walk]:
        df2 = pd.read_csv(path + f"{prefix}_{walk}/{AC}_{walk}_d{str(d)}.csv")
        print(f"Shape {AC}_{walk} = {df2.shape}")
        #print("DF[id]: ", df["id"])
        #print("DF2[id]: ", df2["id"])
        df = pd.concat([df, df2.reindex(df.index)], axis=1)
print("Before dropping duplicates shape: ", df.shape)
NBO_df = df.T.groupby(level=0).first().T
NBO_df = NBO_df.drop_duplicates()
NBO_df = NBO_df.groupby(level=0).first()
print("After dropping duplicates shape: ", NBO_df.shape)
print(NBO_df)
print(f"Shape NBO: {NBO_df.shape}")
NBO_df.to_csv(os.getcwd() + f"/data/autocorrelation_vectors/ABBA_NBO_d{str(d)}.csv")

walks = ["AA"]
"""
Periodic property set (PT) only atom - atom
d - depth
"""
prefix = "PT"
d = 6
path = os.getcwd()
path += "/data/Vaskas_project/"
df = pd.read_csv(path + f"PT_AA/FA_AA_d{str(d)}.csv")
for walk in walks:
    for AC in ACs[walk]:
        df2 = pd.read_csv(path + f"{prefix}_{walk}/{AC}_{walk}_d{str(d)}.csv")
        print(f"Shape {AC}_{walk} = {df2.shape}")
        df = pd.concat([df, df2.reindex(df.index)], axis=1)
print("Before dropping duplicates shape: ", df.shape)
PT_df = df.groupby(level=0).first().T
PT_df= df.T.groupby(level=0).first().T
print("After dropping duplicates shape: ", PT_df.shape)
print(PT_df)
print(f"Shape PT: {PT_df.shape}")
PT_df.to_csv(os.getcwd() + f"/data/autocorrelation_vectors/AA_GP_d{str(d)}.csv")

"""
Electronic property set (NBO) only atom - atom
"""
prefix = "NBO"
path = os.getcwd()
path += "/data/Vaskas_project/"
df = pd.read_csv(path + f"NBO_AA/FA_AA_d{str(d)}.csv")
for walk in walks:
    for AC in ACs[walk]:
        df2 = pd.read_csv(path + f"{prefix}_{walk}/{AC}_{walk}_d{str(d)}.csv")
        print(f"Shape {AC}_{walk} = {df2.shape}")
        #print("DF[id]: ", df["id"])
        #print("DF2[id]: ", df2["id"])
        df = pd.concat([df, df2.reindex(df.index)], axis=1)
print("Before dropping duplicates shape: ", df.shape)
NBO_df = df.T.groupby(level=0).first().T
NBO_df = NBO_df.drop_duplicates()
NBO_df = NBO_df.groupby(level=0).first()
print("After dropping duplicates shape: ", NBO_df.shape)
print(NBO_df)
print(f"Shape NBO: {NBO_df.shape}")
NBO_df.to_csv(os.getcwd() + f"/data/autocorrelation_vectors/AA_NBO_d{str(d)}.csv")


walks = ["BB", "BBavg"]
"""
Periodic property set (PT) only bond-bond
d - depth
"""
prefix = "PT"
d = 6
path = os.getcwd()
path += "/data/Vaskas_project/"
df = pd.read_csv(path + f"PT_BB/FA_BB_d{str(d)}.csv")
for walk in walks:
    for AC in ACs[walk]:
        df2 = pd.read_csv(path + f"{prefix}_{walk}/{AC}_{walk}_d{str(d)}.csv")
        print(f"Shape {AC}_{walk} = {df2.shape}")
        df = pd.concat([df, df2.reindex(df.index)], axis=1)
print("Before dropping duplicates shape: ", df.shape)
PT_df = df.groupby(level=0).first().T
PT_df= df.T.groupby(level=0).first().T
print("After dropping duplicates shape: ", PT_df.shape)
print(PT_df)
print(f"Shape PT: {PT_df.shape}")
PT_df.to_csv(os.getcwd() + f"/data/autocorrelation_vectors/BB_GP_d{str(d)}.csv")

"""
Electronic property set (NBO) only bond - bond
"""
prefix = "NBO"
path = os.getcwd()
path += "/data/Vaskas_project/"
df = pd.read_csv(path + f"NBO_BB/FA_BB_d{str(d)}.csv")
for walk in walks:
    for AC in ACs[walk]:
        df2 = pd.read_csv(path + f"{prefix}_{walk}/{AC}_{walk}_d{str(d)}.csv")
        print(f"Shape {AC}_{walk} = {df2.shape}")
        #print("DF[id]: ", df["id"])
        #print("DF2[id]: ", df2["id"])
        df = pd.concat([df, df2.reindex(df.index)], axis=1)
print("Before dropping duplicates shape: ", df.shape)
NBO_df = df.T.groupby(level=0).first().T
NBO_df = NBO_df.drop_duplicates()
NBO_df = NBO_df.groupby(level=0).first()
print("After dropping duplicates shape: ", NBO_df.shape)
print(NBO_df)
print(f"Shape NBO: {NBO_df.shape}")
NBO_df.to_csv(os.getcwd() + f"/data/autocorrelation_vectors/BB_NBO_d{str(d)}.csv")


walks = ["AB"]
"""
Periodic property set (PT) only bond-bond
d - depth
"""
prefix = "PT"
d = 6
path = os.getcwd()
path += "/data/Vaskas_project/"
df = pd.read_csv(path + f"PT_AB/FA_AB_d{str(d)}.csv")
for walk in walks:
    for AC in ACs[walk]:
        df2 = pd.read_csv(path + f"{prefix}_{walk}/{AC}_{walk}_d{str(d)}.csv")
        print(f"Shape {AC}_{walk} = {df2.shape}")
        df = pd.concat([df, df2.reindex(df.index)], axis=1)
print("Before dropping duplicates shape: ", df.shape)
PT_df = df.groupby(level=0).first().T
PT_df= df.T.groupby(level=0).first().T
print("After dropping duplicates shape: ", PT_df.shape)
print(PT_df)
print(f"Shape PT: {PT_df.shape}")
PT_df.to_csv(os.getcwd() + f"/data/autocorrelation_vectors/AB_GP_d{str(d)}.csv")

"""
Electronic property set (NBO) only bond - bond
"""
prefix = "NBO"
path = os.getcwd()
path += "/data/Vaskas_project/"
df = pd.read_csv(path + f"NBO_AB/FA_AB_d{str(d)}.csv")
for walk in walks:
    for AC in ACs[walk]:
        df2 = pd.read_csv(path + f"{prefix}_{walk}/{AC}_{walk}_d{str(d)}.csv")
        print(f"Shape {AC}_{walk} = {df2.shape}")
        #print("DF[id]: ", df["id"])
        #print("DF2[id]: ", df2["id"])
        df = pd.concat([df, df2.reindex(df.index)], axis=1)
print("Before dropping duplicates shape: ", df.shape)
NBO_df = df.T.groupby(level=0).first().T
NBO_df = NBO_df.drop_duplicates()
NBO_df = NBO_df.groupby(level=0).first()
print("After dropping duplicates shape: ", NBO_df.shape)
print(NBO_df)
print(f"Shape NBO: {NBO_df.shape}")
NBO_df.to_csv(os.getcwd() + f"/data/autocorrelation_vectors/AB_NBO_d{str(d)}.csv")
