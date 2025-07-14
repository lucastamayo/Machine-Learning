import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import pickle
import ROOT
from xgboost import XGBClassifier

# carrega modelo
with open("/eos/user/l/lmonteta/S1_Bc2JPsiTauNu/bdt_model_Bc2JPsiTauNuFull", "rb") as f:
    saved_data = pickle.load(f)

model = saved_data['model']
test_data = saved_data['test_data']
test_labels = saved_data['test_labels']
variables = saved_data['variables']

# carrega pesos
try:
    test_weights = saved_data['test_weights']
except KeyError:
    test_weights = np.ones_like(test_labels)  # Assume peso 1 se não encontrado

#  inferência
test_probs = model.predict_proba(test_data)[:, 1]
test_preds = model.predict(test_data)

#  gerando ROC
fpr, tpr, thresholds = roc_curve(test_labels, test_probs, sample_weight=test_weights)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('/eos/user/l/lmonteta/S1_Bc2JPsiTauNu/roc_curve.png')
plt.show()

#  matriz confusão
cm = confusion_matrix(test_labels, test_preds, sample_weight=test_weights, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Background', 'Signal'])
disp.plot(cmap='Blues')
plt.title('Normalized Confusion Matrix')
plt.savefig('/eos/user/l/lmonteta/S1_Bc2JPsiTauNu/confusion_matrix.png')
plt.show()

#  distribuição de scores
sig_probs = test_probs[test_labels == 1]
bkg_probs = test_probs[test_labels == 0]
sig_weights = test_weights[test_labels == 1]
bkg_weights = test_weights[test_labels == 0]

plt.figure(figsize=(10, 6))
plt.hist(sig_probs, 
         bins=50, 
         range=(0, 1), 
         alpha=0.7, 
         weights=sig_weights, 
         density=True, 
         label='Signal', 
         color='red')
plt.hist(bkg_probs, 
         bins=50, 
         range=(0, 1), 
         alpha=0.7, 
         weights=bkg_weights, 
         density=True, 
         label='Background', 
         color='blue')
plt.xlabel('BDT Score')
plt.ylabel('Normalized Events')
plt.title('BDT Output Distribution')
plt.legend(loc='upper center')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('/eos/user/l/lmonteta/S1_Bc2JPsiTauNu/bdt_scores.png')
plt.show()

#  feature importances
feature_importances = model.feature_importances_
sorted_idx = np.argsort(feature_importances)

plt.figure(figsize=(10, 8))
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(variables)[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Variable Importance')
plt.tight_layout()
plt.savefig('/eos/user/l/lmonteta/S1_Bc2JPsiTauNu/feature_importance.png')
plt.show()
