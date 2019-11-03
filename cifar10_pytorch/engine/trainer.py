import torch

def train(model, device, dataloader, criterion, optimizer, epoch):
    """
    train the model for one epoch.
    Args:
        model: model object.
        device (str): "cuda" or "cpu".
        dataloader: dataloader object.
        criterion: loss function.
        optimizer: optimizer object.
        epoch (int): current epoch number.
    Returns:
        loss (float): loss value.
        acc (float): accuracy value.
    """
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_i > 0 and (batch_i + 1) % 100 == 0:
            print("epoch {}, iter {} / {}, Loss: {:.3f} | Acc: {:.3f}".format(
                epoch+1,
                batch_i+1,
                len(dataloader),
                train_loss/(batch_i+1),
                100.*correct/total
            ))
    return train_loss/(batch_i+1), 100.*correct/total


def test(model, device, dataloader, criterion, epoch):
    """
    Test the model.
    Args:
        model: model object.
        device (str): "cuda" or "cpu".
        dataloader: dataloader object.
        criterion: loss function.
        epoch (int): current epoch number.
    Returns:
        acc (float): accuracy value.
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_i > 0 and (batch_i + 1) % 100 == 0:
                print("epoch {}, iter {} / {}, Loss: {:.3f} | Acc: {:.3f}".format(
                    epoch+1,
                    batch_i+1,
                    len(dataloader),
                    test_loss/(batch_i+1),
                    100.*correct/total,
                ))
    acc = 100.*correct/total
    print("Accuracy: {:.4f}".format(acc))
    return acc