import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import pickle

# Carregando modelo e dados
with open("/eos/user/l/lmonteta/S1_Bc2JPsiTauNuImproved/bdt_model_Bc2JPsiTauNuFull", "rb") as f:
    saved_data = pickle.load(f)

model = saved_data['model']
train_data = saved_data['train_data']
test_data = saved_data['test_data']
train_labels = saved_data['train_labels']
test_labels = saved_data['test_labels']
variables = saved_data['variables']

train_weights = saved_data.get('train_weights', np.ones_like(train_labels))
test_weights = saved_data.get('test_weights', np.ones_like(test_labels))

#  Previsões 
train_probs = model.predict_proba(train_data)[:, 1]
test_probs = model.predict_proba(test_data)[:, 1]


# Curvas ROC

fpr_train, tpr_train, _ = roc_curve(train_labels, train_probs, sample_weight=train_weights)
roc_auc_train = auc(fpr_train, tpr_train)

fpr_test, tpr_test, _ = roc_curve(test_labels, test_probs, sample_weight=test_weights)
roc_auc_test = auc(fpr_test, tpr_test)

plt.figure(figsize=(10, 8))
plt.plot(fpr_train, tpr_train, color='green', lw=2, label=f'Train ROC (AUC = {roc_auc_train:.4f})')
plt.plot(fpr_test, tpr_test, color='darkorange', lw=2, label=f'Test ROC (AUC = {roc_auc_test:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: Train vs Test')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('/eos/user/l/lmonteta/S1_Bc2JPsiTauNuImproved/roc_curve_train_vs_test.png')
plt.close()


# Curvas PR (treino vs teste)

precision_train, recall_train, _ = precision_recall_curve(train_labels, train_probs, sample_weight=train_weights)
ap_train = average_precision_score(train_labels, train_probs, sample_weight=train_weights)

precision_test, recall_test, _ = precision_recall_curve(test_labels, test_probs, sample_weight=test_weights)
ap_test = average_precision_score(test_labels, test_probs, sample_weight=test_weights)

plt.figure(figsize=(10, 8))
plt.step(recall_train, precision_train, where='post', color='green', lw=2, 
         label=f'Train PR (AP = {ap_train:.4f})')
plt.step(recall_test, precision_test, where='post', color='darkorange', lw=2, 
         label=f'Test PR (AP = {ap_test:.4f})')
plt.xlabel('Recall (Signal Efficiency)')
plt.ylabel('Precision (Purity)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('Precision-Recall Curve: Train vs Test')
plt.legend(loc="upper right")
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('/eos/user/l/lmonteta/S1_Bc2JPsiTauNuImproved/pr_curve_train_vs_test.png')
plt.close()


# Distribuição de Score

sig_probs_train = train_probs[train_labels == 1]
bkg_probs_train = train_probs[train_labels == 0]
sig_weights_train = train_weights[train_labels == 1]
bkg_weights_train = train_weights[train_labels == 0]

sig_probs_test = test_probs[test_labels == 1]
bkg_probs_test = test_probs[test_labels == 0]
sig_weights_test = test_weights[test_labels == 1]
bkg_weights_test = test_weights[test_labels == 0]

plt.figure(figsize=(10, 6))
plt.hist(sig_probs_test, bins=50, range=(0, 1), alpha=0.5, weights=sig_weights_test,
         density=True, label='Signal (test)', histtype='stepfilled')
plt.hist(bkg_probs_test, bins=50, range=(0, 1), alpha=0.5, weights=bkg_weights_test,
         density=True, label='Background (test)', histtype='stepfilled')
plt.hist(sig_probs_train, bins=50, range=(0, 1), alpha=0.7, weights=sig_weights_train,
         density=True, label='Signal (train)', histtype='step')
plt.hist(bkg_probs_train, bins=50, range=(0, 1), alpha=0.7, weights=bkg_weights_train,
         density=True, label='Background (train)', histtype='step')
plt.xlabel('BDT Score')
plt.ylabel('Normalized Events')
plt.title('BDT Score Distribution: Train vs Test')
plt.legend(loc='upper center')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('/eos/user/l/lmonteta/S1_Bc2JPsiTauNuImproved/bdt_scores_train_vs_test.png')
plt.close()


# Resumo das métricas

print("\n Métricas ")
print(f"Train ROC AUC: {roc_auc_train:.4f} | Test ROC AUC: {roc_auc_test:.4f}")
print(f"Train PR AUC: {ap_train:.4f} | Test PR AUC: {ap_test:.4f}")
print("\nGráficos comparativos salvos em:")
print("- /eos/user/l/lmonteta/S1_Bc2JPsiTauNuImproved/roc_curve_train_vs_test.png")
print("- /eos/user/l/lmonteta/S1_Bc2JPsiTauNuImproved/pr_curve_train_vs_test.png")
print("- /eos/user/l/lmonteta/S1_Bc2JPsiTauNuImproved/bdt_scores_train_vs_test.png")