import numpy as np
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def precision_and_recall(label_gt, label_pred, n_class):
    assert len(label_gt) == len(label_pred)
    precision = np.zeros(n_class, dtype=np.float32)
    recall = np.zeros(n_class, dtype=np.float32)
    img_A = np.array(label_gt, dtype=np.float32).flatten()
    img_B = np.array(label_pred, dtype=np.float32).flatten()
    precision[:] = precision_score(img_A, img_B, average=None, labels=range(n_class))
    recall[:] = recall_score(img_A, img_B, average=None, labels=range(n_class))

    return precision, recall


def dice_score(label_gt, label_pred, n_class):
    """

    :param label_gt:
    :param label_pred:
    :param n_class:
    :return:
    """

    epsilon = 1.0e-6
    assert len(label_gt) == len(label_pred)
    dice_scores = np.zeros(n_class, dtype=np.float32)
    for class_id in range(n_class):
        img_A = np.array([1 if x == class_id else 0 for x in label_gt], dtype=np.float32)
        img_B = np.array([1 if x == class_id else 0 for x in label_pred], dtype=np.float32)
        # 计算 Dice 系数
        intersection = np.sum(img_A * img_B)  # A 和 B 的交集（True Positives，TP）
        sum_A = np.sum(img_A)  # A 的元素和 (TP + FN)
        sum_B = np.sum(img_B)  # B 的元素和 (TP + FP)

        # 计算每个类别的 Dice 系数
        dice_score = (2.0 * intersection) / (sum_A + sum_B + epsilon)
        dice_scores[class_id] = dice_score

    return dice_scores


def sensitivity_and_specificity(label_gt, label_pred, n_class):
    """
    计算每个类别的 Sensitivity 和 Specificity

    :param label_gt: 真实标签
    :param label_pred: 预测标签
    :param n_class: 类别数
    :return: 每个类别的 Sensitivity 和 Specificity
    """
    epsilon = 1.0e-6
    assert len(label_gt) == len(label_pred)

    sensitivities = np.zeros(n_class, dtype=np.float32)
    specificities = np.zeros(n_class, dtype=np.float32)

    for class_id in range(n_class):
        # 生成当前类别的二值化标签
        img_A = np.array([1 if x == class_id else 0 for x in label_gt], dtype=np.float32)  # Ground truth
        img_B = np.array([1 if x == class_id else 0 for x in label_pred], dtype=np.float32)  # Predictions

        # 计算混淆矩阵中的各个值
        TP = np.sum(img_A * img_B)  # True Positives
        FN = np.sum(img_A * (1 - img_B))  # False Negatives
        TN = np.sum((1 - img_A) * (1 - img_B))  # True Negatives
        FP = np.sum((1 - img_A) * img_B)  # False Positives

        # 计算 Sensitivity 和 Specificity
        sensitivity = TP / (TP + FN + epsilon)
        specificity = TN / (TN + FP + epsilon)

        sensitivities[class_id] = sensitivity
        specificities[class_id] = specificity

    return sensitivities, specificities



def print_comprehensive_score(label_gt, label_pred):
    precision_macro = precision_score(label_gt, label_pred, average='macro')
    recall_macro = recall_score(label_gt, label_pred, average='macro')
    f1_macro = f1_score(label_gt, label_pred, average='macro')

    # 计算混淆矩阵
    cm = confusion_matrix(label_gt, label_pred)

    FP = cm.sum(axis=0) - np.diag(cm)  # 假正例
    FN = cm.sum(axis=1) - np.diag(cm)  # 假负例
    TP = np.diag(cm)  # 真正例
    TN = cm.sum() - (FP + FN + TP)  # 真负例

    epsilon = 1.0e-6

    precision = TP / (TP + FP + epsilon)
    sensitivity = TP / (TP + FN + epsilon)
    specificity = TN / (TN + FP + epsilon)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity + epsilon)

    specificity_macro = np.mean(specificity)

    print(f"Precision (macro): {precision_macro*100:.2f}, Recall/Sensitivity (macro): {recall_macro*100:.2f}\nSpecificity (macro): {specificity_macro*100:.2f}, F1-score (macro): {f1_macro*100:.2f}")

    class_num = 3
    for i in range(class_num):
        print(f"class: {i}, precision: {precision[i]*100:.2f}, sensitivity: {sensitivity[i]*100:.2f}, specificity: {specificity[i]*100:.2f}, F1-score: {f1[i]*100:.2f}")

def print_and_get_comprehensive_score(label_gt, label_pred):
    precision_macro = precision_score(label_gt, label_pred, average='macro')
    recall_macro = recall_score(label_gt, label_pred, average='macro')
    f1_macro = f1_score(label_gt, label_pred, average='macro')

    # 计算混淆矩阵
    cm = confusion_matrix(label_gt, label_pred)

    FP = cm.sum(axis=0) - np.diag(cm)  # 假正例
    FN = cm.sum(axis=1) - np.diag(cm)  # 假负例
    TP = np.diag(cm)  # 真正例
    TN = cm.sum() - (FP + FN + TP)  # 真负例

    epsilon = 1.0e-6

    precision = TP / (TP + FP + epsilon)
    sensitivity = TP / (TP + FN + epsilon)
    specificity = TN / (TN + FP + epsilon)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity + epsilon)

    specificity_macro = np.mean(specificity)

    print(f"Precision (macro): {precision_macro*100:.2f}, Recall/Sensitivity (macro): {recall_macro*100:.2f}\nSpecificity (macro): {specificity_macro*100:.2f}, F1-score (macro): {f1_macro*100:.2f}")

    class_num = 3
    for i in range(class_num):
        print(f"class: {i}, precision: {precision[i]*100:.2f}, sensitivity: {sensitivity[i]*100:.2f}, specificity: {specificity[i]*100:.2f}, F1-score: {f1[i]*100:.2f}")

    return precision_macro, recall_macro, specificity_macro, f1_macro, precision, sensitivity, specificity, f1

