import torch 
import torch.nn as nn
from data_loader import NinaProLoader
from dataset import create_dataloaders
from model import EMGNet
from config import (
    WINDOW_SIZE,
    STEP_SIZE,
    TRAIN_REPS,
    PURITY_THRESHOLD,
    BATCH_SIZE,
    NUM_CLASSES,
    LEARNING_RATE,
    NUM_EPOCHS,
    DROPOUT
)
import os
from torch.utils.tensorboard import SummaryWriter
from model_v2 import EMGNetV2



writer = SummaryWriter(log_dir='runs/emgnet_v2 ')   #emgnet_scheduler


file_path = os.path.join('..', 'data', 'S1_E1_A1.mat')

loader = NinaProLoader(file_path)
loader.load_mat_file()
loader.extract_variables()
loader.normalise(train_reps = TRAIN_REPS)

X_train, y_train, X_test, y_test = loader.create_windows(
    train_reps=TRAIN_REPS,
    purity_threshold=PURITY_THRESHOLD
)

train_loader, test_loader = create_dataloaders(
    X_train, y_train,
    X_test,  y_test,
    batch_size=BATCH_SIZE
)

class_weights = loader.compute_class_weights(y_train, NUM_CLASSES)


model = EMGNetV2() 
print(model)


dummy_input = torch.randn(1, WINDOW_SIZE, 12)
writer.add_graph(model, dummy_input)



criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


def train_one_epoch(model, loader, criterion, optimizer):

    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0


    for X_batch, y_batch in loader:

        predictions = model(X_batch)

        loss = criterion(predictions, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        total_correct += (predictions.argmax(dim=1) == y_batch).sum().item()
        total_samples += len(y_batch)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100
    return avg_loss, accuracy




def evaluate(model, loader, criterion):

    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():

        for X_batch, y_batch in loader:

            predictions = model(X_batch)

            loss = criterion(predictions, y_batch)

            total_loss += loss.item() * len(y_batch)
            total_correct += (predictions.argmax(dim=1) == y_batch).sum().item()
            total_samples += len(y_batch)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100
    return avg_loss, accuracy


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

    #writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], epoch)

    for name, param in model.named_parameters():
        writer.add_histogram(f'weights/{name}', param, epoch)
        if param.grad is not None:
            writer.add_histogram(f'gradients/{name}', param.grad, epoch)


    print(f'{epoch:>6}  {train_loss:>10.4f}  {train_acc:>9.2f}%  {test_loss:>10.4f}  {test_acc:>9.2f}%')

    #scheduler.step()

    # save best model
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), 'best_model.pth')

print(f'\nBest test accuracy: {best_test_acc:.2f}%')
print('Model saved to best_model.pth')

writer.close()