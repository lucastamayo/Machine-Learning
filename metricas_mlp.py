import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import pickle
from tensorflow.keras.models import load_model

#  Carregando modelo e dados 
mlp_model = load_model("/eos/user/l/lmonteta/S1_Bc2JPsiTauNuImproved/mlp_model.h5")

with open("/eos/user/l/lmonteta/S1_Bc2JPsiTauNuImproved/mlp_model_Bc2JPsiTauNuFull", "rb") as f:
    saved_data = pickle.load(f)

X_test = saved_data['test_data']
y_test = saved_data['test_labels']
w_test = saved_data.get('test_weights', np.ones_like(y_test))
variables = saved_data['variables']

#  Inferência 
probs = mlp_model.predict(X_test).ravel()
preds = (probs > 0.5).astype(int)

#  ROC Curve 
fpr, tpr, thresholds = roc_curve(y_test, probs, sample_weight=w_test)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('MLP ROC Curve')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('/eos/user/l/lmonteta/S1_Bc2JPsiTauNuImproved/mlp_roc_curve.png')
# plt.show()

#  matriz confusão
cm = confusion_matrix(y_test, preds, sample_weight=w_test, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Background', 'Signal'])
disp.plot(cmap='Blues', values_format=".1f")
plt.title('MLP Normalized Confusion Matrix (%)')
plt.savefig('/eos/user/l/lmonteta/S1_Bc2JPsiTauNuImproved/mlp_confusion_matrix.png')
# plt.show()

#  Distribuição do Score 
sig_probs = probs[y_test == 1]
bkg_probs = probs[y_test == 0]
sig_weights = w_test[y_test == 1]
bkg_weights = w_test[y_test == 0]

plt.figure(figsize=(10, 6))
plt.hist(sig_probs, bins=50, range=(0, 1), alpha=0.7, weights=sig_weights,
         density=True, label='Signal', color='red')
plt.hist(bkg_probs, bins=50, range=(0, 1), alpha=0.7, weights=bkg_weights,
         density=True, label='Background', color='blue')
plt.xlabel('MLP Output')
plt.ylabel('Normalized Events')
plt.title('MLP Score Distribution')
plt.legend(loc='upper center')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('/eos/user/l/lmonteta/S1_Bc2JPsiTauNuImproved/mlp_scores.png')
# plt.show()

