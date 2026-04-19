import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from data_loader import NinaProLoader, load_multiple_subjects
from dataset import create_dataloaders
from model import EMGNet
from model_v2 import EMGNetV2
from config import (
    WINDOW_SIZE, TRAIN_REPS, PURITY_THRESHOLD,
    BATCH_SIZE, NUM_CLASSES
)


# ─── config ─────────────────────────────────────────
MODEL_PATH = 'best_model.pth'
USE_V2 = False   # set to True if your saved model is V2


# ─── device setup ───────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}\n')


# ─── load data (same 5 subjects) ────────────────────
file_paths = [
    '/content/sample_data/S1_E1_A1.mat',
    '/content/sample_data/S2_E1_A1.mat',
    '/content/sample_data/S3_E1_A1.mat',
    '/content/sample_data/S4_E1_A1.mat',
    '/content/sample_data/S5_E1_A1.mat',
]

X_train, y_train, X_test, y_test = load_multiple_subjects(
    file_paths,
    train_reps=TRAIN_REPS,
    purity_threshold=PURITY_THRESHOLD
)

_, test_loader = create_dataloaders(
    X_train, y_train, X_test, y_test, batch_size=BATCH_SIZE
)

# class weights (match what was used during training)
loader = NinaProLoader(file_paths[0])
class_weights = loader.compute_class_weights(y_train, NUM_CLASSES).to(device)


# ─── load trained model ─────────────────────────────
if USE_V2:
    model = EMGNetV2().to(device)
    print('Loading V2 model...\n')
else:
    model = EMGNet().to(device)
    print('Loading simple model...\n')

model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

criterion = nn.CrossEntropyLoss(weight=class_weights)


# ─── gesture names for readability ──────────────────
GESTURE_NAMES = {
    0:  'Rest',
    1:  'Thumb up',
    2:  'Index + middle ext',
    3:  'Ring + little flex',
    4:  'Thumb opposing',
    5:  'Fingers flexed',
    6:  'Fist',
    7:  'Pointing index',
    8:  'Adducted fingers',
    9:  'Wrist supination (mid)',
    10: 'Wrist pronation (mid)',
    11: 'Wrist supination (little)',
    12: 'Wrist pronation (little)',
    13: 'Wrist flexion',
    14: 'Wrist extension',
    15: 'Wrist radial deviation',
    16: 'Wrist ulnar deviation',
    17: 'Wrist extended + fist',
}


# ─── run evaluation with confusion matrix ──────────
def analyse(model, loader, criterion, num_classes):
    model.eval()
    per_class_correct = torch.zeros(num_classes)
    per_class_total   = torch.zeros(num_classes)
    total_loss, total_samples = 0, 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            predicted = predictions.argmax(dim=1)
            total_loss += loss.item() * len(y_batch)
            total_samples += len(y_batch)

            y_cpu = y_batch.cpu()
            pred_cpu = predicted.cpu()

            all_preds.extend(pred_cpu.numpy())
            all_labels.extend(y_cpu.numpy())

            for cls in range(num_classes):
                mask = y_cpu == cls
                per_class_total[cls] += mask.sum().item()
                per_class_correct[cls] += ((pred_cpu == y_cpu) & mask).sum().item()


    # ─── per class accuracy ─────────────────────────
    print('=== Per-class accuracy ===')
    for cls in range(num_classes):
        if per_class_total[cls] > 0:
            acc = per_class_correct[cls] / per_class_total[cls] * 100
            name = GESTURE_NAMES.get(cls, '?')
            print(f'  class {cls:2d} ({name:28s}): {acc:6.2f}%  ({int(per_class_correct[cls])}/{int(per_class_total[cls])})')

    class_accs = (per_class_correct / per_class_total.clamp(min=1)) * 100
    valid = per_class_total > 0
    print(f'\nMean class accuracy: {class_accs[valid].mean().item():.2f}%')
    print(f'Std class accuracy:  {class_accs[valid].std().item():.2f}%')
    print(f'Avg test loss:       {total_loss/total_samples:.4f}')


    # ─── confusion matrix ───────────────────────────
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))


    # ─── analyse worst classes ──────────────────────
    print('\n\n=== CONFUSION ANALYSIS FOR WORST CLASSES ===')
    worst_classes = [13, 16]

    for worst_class in worst_classes:
        row = cm[worst_class]
        total = row.sum()
        if total == 0:
            continue

        worst_name = GESTURE_NAMES.get(worst_class, '?')
        print(f'\n--- When TRUE class is {worst_class} ({worst_name}), model predicts: ---')
        sorted_indices = np.argsort(row)[::-1]
        for cls in sorted_indices:
            if row[cls] > 0:
                pct = row[cls] / total * 100
                pred_name = GESTURE_NAMES.get(cls, '?')
                marker = '  ← CORRECT' if cls == worst_class else ''
                print(f'  class {cls:2d} ({pred_name:28s}): {int(row[cls]):4d} times ({pct:5.1f}%){marker}')


    # ─── full confusion matrix table ────────────────
    print('\n\n=== FULL CONFUSION MATRIX (row = true, col = predicted) ===')
    print('Values are % of true class (row sums to ~100)')
    print('      ' + ' '.join([f'{i:4d}' for i in range(num_classes)]))
    for i in range(num_classes):
        row_pct = cm[i] / cm[i].sum() * 100 if cm[i].sum() > 0 else cm[i]
        row_str = ' '.join([f'{v:4.0f}' if v >= 1 else '   .' for v in row_pct])
        print(f'  {i:2d}  {row_str}')


    # ─── visual confusion matrix ────────────────────
    print('\n\nCreating confusion matrix heatmap...')
    fig, ax = plt.subplots(figsize=(12, 10))
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    cm_pct = np.nan_to_num(cm_pct)

    im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)

    for i in range(num_classes):
        for j in range(num_classes):
            if cm_pct[i, j] > 5:
                color = 'white' if cm_pct[i, j] > 50 else 'black'
                ax.text(j, i, f'{cm_pct[i, j]:.0f}', ha='center', va='center',
                        color=color, fontsize=9)

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(range(num_classes), fontsize=10)
    ax.set_yticklabels(range(num_classes), fontsize=10)
    ax.set_xlabel('Predicted class', fontsize=12)
    ax.set_ylabel('True class', fontsize=12)
    ax.set_title('Confusion Matrix (% of true class)', fontsize=14, pad=12)

    # highlight worst classes with red border
    for worst_class in [13, 16]:
        ax.add_patch(plt.Rectangle((-0.5, worst_class-0.5), num_classes, 1,
                                    fill=False, edgecolor='red', linewidth=2))

    plt.colorbar(im, ax=ax, label='% of true class')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print('Saved: confusion_matrix.png')
    plt.show()


if __name__ == '__main__':
    analyse(model, test_loader, criterion, NUM_CLASSES)