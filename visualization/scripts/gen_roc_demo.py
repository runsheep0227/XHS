"""One-off: emit compact demo eval_viz JSON + print AUCs (sklearn)."""
import json
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# 弱可分：AUC 约 0.84–0.87，接近 comment/results/06_roc_curves.png 量级
np.random.seed(11)
n = 1200
y = np.random.choice([0, 1, 2], size=n, p=[0.31, 0.36, 0.33])
P = np.random.dirichlet([1.6, 1.6, 1.6], size=n).astype(np.float64)
for c in range(3):
    m = y == c
    P[m, c] *= 2.05
    P[m] /= P[m].sum(axis=1, keepdims=True)
for _ in range(2):
    for c in range(3):
        m = y == c
        P[m, c] += 0.07
        P[m] /= P[m].sum(axis=1, keepdims=True)
Y = label_binarize(y, classes=[0, 1, 2])
for i in range(3):
    fpr, tpr, _ = roc_curve(Y[:, i], P[:, i])
    print(i, "auc", round(auc(fpr, tpr), 4))
fpr_m, tpr_m, _ = roc_curve(Y.ravel(), P.ravel())
print("micro", round(auc(fpr_m, tpr_m), 4))
pred = P.argmax(axis=1)
payload = {
    "true_labels": y.astype(int).tolist(),
    "pred_labels": pred.astype(int).tolist(),
    "probs": P.tolist(),
}
import os

_here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(_here, "..", "src", "data", "evalVizDemoPayload.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(payload, f, separators=(",", ":"))
print("write", out, "n=", len(payload["true_labels"]))
