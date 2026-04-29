import torch
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter
from data_loader import NinaProLoader, load_multiple_subjects
from dataset import create_dataloaders
from model import EMGNet
from model_v2 import EMGNetV2
from config import (
    WINDOW_SIZE,
    STEP_SIZE,
    TRAIN_REPS,
    PURITY_THRESHOLD,
    BATCH_SIZE,
    NUM_CLASSES,
    LEARNING_RATE,
    NUM_EPOCHS,
    DROPOUT,
    NUM_CHANNELS
)


"""HEADS UP: I had to run the test accuracy only on one subject becaue the RAM did not have enough space to 
allow we to find test accuracies on each subject during training, so I made test_subjects.py that runs the model 
against all the subjects ( both train and test data) to see the general results of my model."""


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


writer = SummaryWriter(log_dir='runs/emgnet_5subjects')



file_paths = [
    os.path.join('..', 'data', 'S1_E1_A1.mat'),
    os.path.join('..', 'data', 'S2_E1_A1.mat'),
    os.path.join('..', 'data', 'S3_E1_A1.mat'),
    os.path.join('..', 'data', 'S6_E1_A1.mat'),
    os.path.join('..', 'data', 'S5_E1_A1.mat'),
    os.path.join('..', 'data', 'S11_E1_A2.mat'),
    os.path.join('..', 'data', 'S1_E1_A2.mat'),
]

X_train, y_train, X_test, y_test = load_multiple_subjects(
    file_paths,
    train_reps=TRAIN_REPS,
    purity_threshold=PURITY_THRESHOLD
)

train_loader, test_loader = create_dataloaders(
    X_train, y_train,
    X_test,  y_test,
    batch_size=BATCH_SIZE
)

# recompute class weights on combined data
loader = NinaProLoader(file_paths[0])
class_weights = loader.compute_class_weights(y_train, NUM_CLASSES).to(device)



model = EMGNet().to(device)
#model = EMGNetV2().to(device)
print(model)

dummy_input = torch.randn(1, WINDOW_SIZE, NUM_CHANNELS).to(device)
writer.add_graph(model, dummy_input)


criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss    = 0
    total_correct = 0
    total_samples = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        predictions = model(X_batch)
        loss        = criterion(predictions, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss    += loss.item() * len(y_batch)
        total_correct += (predictions.argmax(dim=1) == y_batch).sum().item()
        total_samples += len(y_batch)

    return total_loss / total_samples, total_correct / total_samples * 100


def evaluate(model, loader, criterion):
    model.eval()
    total_loss    = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            loss        = criterion(predictions, y_batch)

            total_loss    += loss.item() * len(y_batch)
            total_correct += (predictions.argmax(dim=1) == y_batch).sum().item()
            total_samples += len(y_batch)

    return total_loss / total_samples, total_correct / total_samples * 100


def evaluate_detailed(model, loader, criterion, num_classes):
    model.eval()
    per_class_correct = torch.zeros(num_classes)
    per_class_total   = torch.zeros(num_classes)
    total_loss = 0
    total_samples = 0

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

            for cls in range(num_classes):
                mask = y_cpu == cls
                per_class_total[cls] += mask.sum().item()
                per_class_correct[cls] += ((pred_cpu == y_cpu) & mask).sum().item()

    print('\n=== Per-class accuracy ===')
    for cls in range(num_classes):
        if per_class_total[cls] > 0:
            acc = per_class_correct[cls] / per_class_total[cls] * 100
            print(f'  class {cls:2d}: {acc:6.2f}%  ({int(per_class_correct[cls])}/{int(per_class_total[cls])})')

    class_accs = (per_class_correct / per_class_total.clamp(min=1)) * 100
    valid = per_class_total > 0
    mean_acc = class_accs[valid].mean().item()
    std_acc  = class_accs[valid].std().item()

    print(f'\nMean class accuracy:  {mean_acc:.2f}%')
    print(f'Std class accuracy:   {std_acc:.2f}%    ← lower = more consistent')
    print(f'Avg test loss:        {total_loss/total_samples:.4f}')


print(f'\nTraining for {NUM_EPOCHS} epochs...\n')
print(f'{"epoch":>6}  {"train loss":>10}  {"train acc":>10}  {"test loss":>10}  {"test acc":>10}')
print('-' * 56)

best_test_acc = 0.0



for epoch in range(1, NUM_EPOCHS + 1):

    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    test_loss,  test_acc  = evaluate(model, test_loader, criterion)

    writer.add_scalars('loss', {
        'train': train_loss,
        'test':  test_loss
    }, epoch)

    writer.add_scalars('accuracy', {
        'train': train_acc,
        'test':  test_acc
    }, epoch)

    for name, param in model.named_parameters():
        writer.add_histogram(f'weights/{name}', param, epoch)
        if param.grad is not None:
            writer.add_histogram(f'gradients/{name}', param.grad, epoch)

    print(f'{epoch:>6}  {train_loss:>10.4f}  {train_acc:>9.2f}%  {test_loss:>10.4f}  {test_acc:>9.2f}%')

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), 'best_model.pth')

print(f'\nBest test accuracy: {best_test_acc:.2f}%')
print('Model saved to best_model.pth')

# load best model and run detailed evaluation
print('\n\n=== DETAILED EVALUATION ON BEST MODEL ===')
model.load_state_dict(torch.load('best_model.pth'))
evaluate_detailed(model, test_loader, criterion, NUM_CLASSES)

writer.close()