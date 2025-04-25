import pandas as pd

uids = df[["uid"]]
history = pd.merge(uids, self.data_loader.train_his_df, on="uid", how="left")
history = history.rename(columns={"iids": global_p.C_HISTORY})
self.args.device
paddle.max
