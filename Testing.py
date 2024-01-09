import os
import yaml
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
from models.data_loader import build_data_loader
from models.lstm import build_lstm
from Evaluation import *


def calculate_binary_classifier(gt, hypo):
    TP, FP, FN = 0, 0, 0
    for idx, x in enumerate(gt):
        #print(x, hypo[idx])
        if x == 1 and round(hypo[idx]) == 1:
            TP += 1
        elif x == 0 and round(hypo[idx]) == 1:
            FP += 1
        elif x == 1 and round(hypo[idx]) == 0:
            FN += 1
    return TP, FP, FN


@torch.no_grad()
def test():
    # Load config file
    with open('configuration.yaml', 'r') as file:
        config = yaml.safe_load(file)

    model_dir = config['training']['saved_dir']
    model_file = config['training']['saved_best_model']

    model = torch.load(os.path.join(model_dir, model_file))

    model.eval()

    # Load test data_loader
    test_loader = build_data_loader(config, 'test')

    gts = []
    preds = []
    elapsed_time = []

    # Shape of inputs: (batch_size, sequence_length, num_features)
    # Shape of targets: (batch_size, sequence_length, binary_label)
    for idx, (inputs, targets) in enumerate(test_loader):
        start_time = time.time()

        pred = model(inputs)

        elapsed_time.append(time.time() - start_time)
        #print("Out shape: ", pred.shape)
        #print(pred, targets)
        preds.append(float(torch.sigmoid(pred[0][0]).detach()))

        gts.append(float(targets[0][0].detach()))

    print("Inference mean latency = %.02f ms" % (sum(elapsed_time)/len(elapsed_time) * 1000))

    print(gts[:20], preds[:20])
    # aggregate
    tp, fp, fn = calculate_binary_classifier(gts, preds)
    print("TP = {}, FP = {}, FN = {}, among a total of {} testing samples.".format(tp, fp, fn, len(test_loader)))
    print()
    precision = calculate_precision(tp, fp)
    recall = calculate_recall(tp, fn)
    F1 = calculate_f1_score(precision, recall)

    print("Precision = {}, Recall = {}, F1 score = {}".format(precision, recall, F1))

    FPr, TPr, thresholds = roc_curve(gts, preds, sample_weight=None, drop_intermediate=True)

    # Calculate the AUC score
    roc_auc = auc(FPr, TPr)
    print("AUC = {}".format(roc_auc))

    # Plotting the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(FPr, TPr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')

    plt.savefig(os.path.join('./imgs', 'roc_curve2.pdf'), bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    test()
