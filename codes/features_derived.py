#!/usr/bin/env python
# encoding: utf-8
import os
import sys
import json
import numpy as np
import pandas as pd
import _pickle as pk
from scipy.stats import mode
from ai_hub import inferServer

real = os.path.exists("/tcdata")
hist = None
hpoi = 0
hpos = []
fail = set()
t1 = 0.4 if real else 0
t2 = 0.9

# 加载模型，需要返回一个模型带predict方法的模型，否则要重写下面的predict方法
def load_model():
    with open("model.pkl", "rb") as f:
        model = pk.load(f)
    return model


# 特征提取，需要返回一个DataFrame，第一列为serial_number，第二列开始为特征
def extract_feature(mce, adr, krn):
    global hist, hpoi, hpos
    mce["mca_id"] = mce["mca_id"].fillna("NA")
    mce["transaction"] = mce["transaction"].fillna(4)
    mce["vendor"] = mce["vendor"].fillna(3)
    npmce = mce.values
    dat = []
    mca = [
        "Z",
        "NA",
        "AP",
        "AF",
        "E",
        "CD",
        "BB",
        "C",
        "CC",
        "F",
        "G",
        "EE",
        "AA",
        "AE",
        "BC",
        "AZ",
        "DE",
        "FF",
    ]
    for s in np.unique(npmce[:, 0]):
        sdf = npmce[npmce[:, 0] == s]
        dat.append([s, sdf.shape[0], sdf[:, 4][0], sdf[:, 5][0]])
        for i in mca:
            dat[-1].append(sdf[sdf[:, 1] == i].shape[0])
        for t in range(5):
            dat[-1].append(sdf[sdf[:, 2] == t].shape[0])
    ces = pd.DataFrame(
        dat,
        columns=[
            "SN",
            "CT",
            "MF",
            "VD",
            "Z",
            "NA",
            "AP",
            "AF",
            "E",
            "CD",
            "BB",
            "C",
            "CC",
            "F",
            "G",
            "EE",
            "AA",
            "AE",
            "BC",
            "AZ",
            "DE",
            "FF",
            "T0",
            "T1",
            "T2",
            "T3",
            "T4",
        ],
    )
    ces["VD"] = ces["VD"].astype("int64")
    npadr = adr.values
    dat = []
    for s in np.unique(npadr[:, 0]):
        sdf = npadr[npadr[:, 0] == s]
        dat.append([s])
        for i in range(1, 6):
            dat[-1].extend(
                [
                    mode(sdf[:, i]).mode[0],
                    mode(sdf[:, i]).count[0],
                    np.std(sdf[:, i]),
                ]
            )
    drs = pd.DataFrame(
        dat,
        columns=[
            "SN",
            "M1",
            "C1",
            "S1",
            "M2",
            "C2",
            "S2",
            "M3",
            "C3",
            "S3",
            "M4",
            "C4",
            "S4",
            "M5",
            "C5",
            "S5",
        ],
    )
    krn.fillna(0, inplace=True)
    npkrn = krn.values
    dat = []
    for s in np.unique(npkrn[:, 25]):
        sdf = npkrn[npkrn[:, 25] == s]
        dat.append([s])
        dat[-1].extend(np.sum(sdf[:, 1:25], axis=0).tolist())
    cols = ["SN"]
    cols.extend(["K" + str(i) for i in range(1, 25)])
    rns = pd.DataFrame(dat, columns=cols)
    rns.fillna(0, inplace=True)
    rns[cols[1:]] = rns[cols[1:]].astype("int64")
    full = pd.merge(ces, drs, "right", on=["SN"])
    full = pd.merge(full, rns, "left", on=["SN"])
    full.fillna(0, inplace=True)
    full.reset_index(drop=True, inplace=True)
    full.iloc[:, 1:] = full.iloc[:, 1:].astype("float32")
    if hist is None:
        hist = pd.DataFrame([], columns=full.columns)
    hist = hist.append(full)
    hpoi += 1
    hpos.append(full.shape[0])
    if hpoi > 1024:
        hist = hist.iloc[hpos[hpoi - 1025] :]
    daily = hist.groupby("SN").sum().reset_index()
    full = pd.merge(full, daily, how="left", on="SN", suffixes=("", "10")).fillna(0)
    return full


class myInfer(inferServer):
    def __init__(self, model):
        super().__init__(model)

    # 数据预处理
    def pre_process(self, request):
        json_data = request.get_json()
        try:
            mce_log = pd.DataFrame(
                json_data["mce_log"],
                columns=[
                    "serial_number",
                    "mca_id",
                    "transaction",
                    "collect_time",
                    "manufacturer",
                    "vendor",
                ],
            )
        except:
            mce_log = pd.DataFrame(
                [],
                columns=[
                    "serial_number",
                    "mca_id",
                    "transaction",
                    "collect_time",
                    "manufacturer",
                    "vendor",
                ],
            )
        try:
            address_log = pd.DataFrame(
                json_data["address_log"],
                columns=[
                    "serial_number",
                    "memory",
                    "rankid",
                    "bankid",
                    "row",
                    "col",
                    "collect_time",
                    "manufacturer",
                    "vendor",
                ],
            )
        except:
            address_log = pd.DataFrame(
                [],
                columns=[
                    "serial_number",
                    "memory",
                    "rankid",
                    "bankid",
                    "row",
                    "col",
                    "collect_time",
                    "manufacturer",
                    "vendor",
                ],
            )
        try:
            kernel_log = pd.DataFrame(
                json_data["kernel_log"],
                columns=[
                    "collect_time",
                    "1_hwerr_f",
                    "1_hwerr_e",
                    "2_hwerr_c",
                    "2_sel",
                    "3_hwerr_n",
                    "2_hwerr_s",
                    "3_hwerr_m",
                    "1_hwerr_st",
                    "1_hw_mem_c",
                    "3_hwerr_p",
                    "2_hwerr_ce",
                    "3_hwerr_as",
                    "1_ke",
                    "2_hwerr_p",
                    "3_hwerr_kp",
                    "1_hwerr_fl",
                    "3_hwerr_r",
                    "_hwerr_cd",
                    "3_sup_mce_note",
                    "3_cmci_sub",
                    "3_cmci_det",
                    "3_hwerr_pi",
                    "3_hwerr_o",
                    "3_hwerr_mce_l",
                    "serial_number",
                    "manufacturer",
                    "vendor",
                ],
            )
        except:
            kernel_log = pd.DataFrame(
                [],
                columns=[
                    "collect_time",
                    "1_hwerr_f",
                    "1_hwerr_e",
                    "2_hwerr_c",
                    "2_sel",
                    "3_hwerr_n",
                    "2_hwerr_s",
                    "3_hwerr_m",
                    "1_hwerr_st",
                    "1_hw_mem_c",
                    "3_hwerr_p",
                    "2_hwerr_ce",
                    "3_hwerr_as",
                    "1_ke",
                    "2_hwerr_p",
                    "3_hwerr_kp",
                    "1_hwerr_fl",
                    "3_hwerr_r",
                    "_hwerr_cd",
                    "3_sup_mce_note",
                    "3_cmci_sub",
                    "3_cmci_det",
                    "3_hwerr_pi",
                    "3_hwerr_o",
                    "3_hwerr_mce_l",
                    "serial_number",
                    "manufacturer",
                    "vendor",
                ],
            )
        if address_log.shape[0] != 0:
            test_data = extract_feature(mce_log, address_log, kernel_log)
            return test_data
        else:
            return None

    # 数据后处理
    def post_process(self, data):
        if data.shape[0] == 0:
            if not real:
                print("[]", file=sys.stderr)
            return "[]"
        data.columns = ["serial_number", "pti"]
        ret = data.to_json(orient="records")
        if not real:
            print(ret, file=sys.stderr)
        print(f"Total bad servers: {len(fail)}", file=sys.stderr)
        return ret

    # 预测方法，按需重写
    def predict(self, data):
        global fail
        if data is not None:
            ret = np.zeros((data.shape[0], 10))
            for i in range(10):
                ret[:, i] = self.model[i].predict_proba(data.iloc[:, 1:].values)[:, 1]
            data["pti"] = np.mean(ret, axis=1)
            data = data[data["pti"] > t1][["SN", "pti"]].reset_index(drop=True)
            if data.shape[0] > 0:
                for i in range(data.shape[0]):
                    if (data["SN"][i] in fail) and (data["pti"][i] < t2):
                        data.iloc[i, 1] = 0
                    else:
                        fail.add(data["SN"][i])
                        if real:
                            data.iloc[i, 1] = 5
            return data[data["pti"] > t1]
        else:
            print("No predictable samples!", file=sys.stderr)
            return pd.DataFrame()


if __name__ == "__main__":
    mymodel = load_model()
    my_infer = myInfer(mymodel)
    my_infer.run(debuge=False)
