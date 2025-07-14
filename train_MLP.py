import uproot
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks


def load_process(fIn, variables, target=0, weight_sf=1.):
    f = uproot.open(fIn)
    tree = f["events"]
    weight = 1.0 / tree.num_entries * weight_sf
    print(f"Carrega {fIn} com {tree.num_entries} eventos e peso {weight}")

    arrs = tree.arrays(variables, library="np")
    df = pd.DataFrame({
        key: val for key, val in arrs.items()
        if isinstance(val, np.ndarray) and val.ndim == 1 and not isinstance(val[0], (list, dict, np.ndarray))
    })


    df["target"] = target
    df["weight"] = weight
    return df

# Variáveis utilizadas
variables = ["q2", "BcMass", "BcEnergy"]

# Caminhos dos arquivos
signal_path = "/eos/user/l/lmonteta/S1_Bc2JPsiTauNu_BDT_Improved_Signal/p8_ee_Zbb_ecm91_EvtGen_Bc2JPsiTauNuImproved.root"
bkg_path = "/eos/user/l/lmonteta/S1_Bc2JPsiTauNu_BDT_Improved_Background/p8_ee_Zbb_ecm91_EvtGen_Bc2JPsiTauNuImproved.root"

# Normalização dos pesos
weight_sf = 1e9
sig_df = load_process(signal_path, variables, target=1, weight_sf=weight_sf)
bkg_df = load_process(bkg_path, variables, target=0, weight_sf=weight_sf)

# Junta os dados
data = pd.concat([sig_df, bkg_df], ignore_index=True)
X = data[variables].to_numpy()
y = data["target"].to_numpy()
w = data["weight"].to_numpy()

# Cross-validation manual
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_aucs = []

print("\n--- Iniciando Cross-Validation com MLP ---")
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n[Fold {fold+1}]")

    X_train_cv, X_val_cv = X[train_idx], X[val_idx]
    y_train_cv, y_val_cv = y[train_idx], y[val_idx]
    w_train_cv, w_val_cv = w[train_idx], w[val_idx]

    model = keras.Sequential([
        layers.Input(shape=(X.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[keras.metrics.AUC(name="auc")]
    )

    early_stop = callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_auc", mode="max")

    model.fit(
        X_train_cv, y_train_cv,
        sample_weight=w_train_cv,
        validation_data=(X_val_cv, y_val_cv, w_val_cv),
        epochs=100,
        batch_size=256,
        verbose=0,
        callbacks=[early_stop]
    )

    y_pred_val = model.predict(X_val_cv).ravel()
    auc_val = roc_auc_score(y_val_cv, y_pred_val, sample_weight=w_val_cv)
    print(f"AUC = {auc_val:.4f}")
    cv_aucs.append(auc_val)

print("\nCross-validated AUCs:", cv_aucs)
print("Mean AUC:", np.mean(cv_aucs))

# Divisão treino/teste final
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, w, test_size=0.2, random_state=42, stratify=y
)

print("\nTreinando modelo final MLP")
final_model = keras.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

final_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[keras.metrics.AUC(name="auc")]
)

early_stop = callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_auc", mode="max")

final_model.fit(
    X_train, y_train,
    sample_weight=w_train,
    validation_data=(X_test, y_test, w_test),
    epochs=100,
    batch_size=256,
    verbose=1,
    callbacks=[early_stop]
)

# Salvando modelo
h5_path = "/eos/user/l/lmonteta/S1_Bc2JPsiTauNuImproved/mlp_model.h5"
final_model.save(h5_path)


pickle_path = "/eos/user/l/lmonteta/S1_Bc2JPsiTauNuImproved/mlp_model_Bc2JPsiTauNuFull"
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

