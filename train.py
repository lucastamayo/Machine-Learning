import uproot
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
import ROOT
import pickle

ROOT.gROOT.SetBatch(True)

def load_process(fIn, variables, target=0, weight_sf=1.):
    f = uproot.open(fIn)
    tree = f["events"]
    weight = 1.0 / tree.num_entries * weight_sf
    print(f"carrega {fIn} com {tree.num_entries} eventos and peso {weight}")

    arrs = tree.arrays(variables, library="np")
    df = pd.DataFrame({
        key: val for key, val in arrs.items()
        if isinstance(val, np.ndarray) and val.ndim == 1 and not isinstance(val[0], (list, dict, np.ndarray))
    })


    df["target"] = target
    df["weight"] = weight
    return df

# Variáveis utilizadas no BDT
variables = ["q2", "BcMass", "BcEnergy"]

# Caminhos dos arquivos
signal_path = "/eos/user/l/lmonteta/S1_Bc2JPsiTauNu_BDT_Improved_Signal/p8_ee_Zbb_ecm91_EvtGen_Bc2JPsiTauNuImproved.root"
bkg_path = "/eos/user/l/lmonteta/S1_Bc2JPsiTauNu_BDT_Improved_Background/p8_ee_Zbb_ecm91_EvtGen_Bc2JPsiTauNuImproved.root"

# Normalização dos pesos
weight_sf = 1e9
sig_df = load_process(signal_path, variables, target=1, weight_sf=weight_sf)
bkg_df = load_process(bkg_path, variables, target=0, weight_sf=weight_sf)

# concatenação dos dados
data = pd.concat([sig_df, bkg_df], ignore_index=True)

X = data[variables].to_numpy()
y = data["target"].to_numpy()
w = data["weight"].to_numpy()

# hiperparâmetros
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.05,
    'max_depth': 4,
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'gamma': 3,
    'min_child_weight': 15,
    'n_estimators': 1000,
    'seed': 42,
}

# Cross-validation com pesos
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_aucs = []

print("\n Iniciando Cross-Validation")
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n[Fold {fold+1}]")
    X_train_cv, X_val_cv = X[train_idx], X[val_idx]
    y_train_cv, y_val_cv = y[train_idx], y[val_idx]
    w_train_cv, w_val_cv = w[train_idx], w[val_idx]

    bdt = xgb.XGBClassifier(**params)
    bdt.fit(
        X_train_cv, y_train_cv,
        sample_weight=w_train_cv,
        eval_set=[(X_val_cv, y_val_cv)],
        sample_weight_eval_set=[w_val_cv],
        verbose=False
    )

    y_pred = bdt.predict_proba(X_val_cv)[:, 1]
    auc = roc_auc_score(y_val_cv, y_pred, sample_weight=w_val_cv)
    print(f"AUC = {auc:.4f}")
    cv_aucs.append(auc)

print("\nCross-validated AUCs:", cv_aucs)
print("Mean AUC:", np.mean(cv_aucs))

# Divisão treino/teste 
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, w, test_size=0.2, random_state=42, stratify=y
)

print("\nTreinando modelo final no conjunto de treino")
final_model = xgb.XGBClassifier(**params)
final_model.fit(
    X_train, y_train,
    sample_weight=w_train,
    eval_set=[(X_test, y_test)],
    sample_weight_eval_set=[w_test],
    verbose=True,
)

# Salvando modelo no ROOT
fOutName = "/eos/user/l/lmonteta/S1_Bc2JPsiTauNuImproved/bdt_model_Bc2JPsiTauNuFull.root"
ROOT.TMVA.Experimental.SaveXGBoost(final_model, "bdt_model", fOutName, num_inputs=len(variables))

# Salvando variáveis 
variables_ = ROOT.TList()
for var in variables:
    variables_.Add(ROOT.TObjString(var))
fOut = ROOT.TFile(fOutName, "UPDATE")
fOut.WriteObject(variables_, "variables")

# Salvando modelo completo 
pickle_path = "/eos/user/l/lmonteta/S1_Bc2JPsiTauNuImproved/bdt_model_Bc2JPsiTauNuFull"
save = {
    'model': final_model,
    'train_data': X_train,
    'test_data': X_test,
    'train_labels': y_train,
    'test_labels': y_test,
    'train_weights': w_train,
    'test_weights': w_test,
    'variables': variables
}
pickle.dump(save, open(pickle_path, "wb"))

print(f"\nModelo salvo em: {pickle_path}")
