import os
import csv

import torch

from ignite.metrics import Accuracy, Loss
from ignite.engine import Events, create_supervised_evaluator
from ignite.engine import create_supervised_trainer
import time


def test_model(model, testloader, device):
    """
    Function to test a model. Returns percentage accuracy.
    """
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def train_model(
        model, trainloader, testloader, optimizer,
        criterion, num_epochs, device,
        output_directory=None, retrain=False):
    """
    Function to train a model.
    """
    log_output = []
    final_accuracy = 0.0
    for epoch in range(num_epochs):
        if epoch == 0:
            accuracy = test_model(model, testloader, device)
            print("ep  {:03d}  loss    {:.3f}  acc  {:.3f}%".format(epoch,
                  0, accuracy))
            final_accuracy = accuracy

        mini_batch_loss = 0
        for _, data in enumerate(trainloader, 0):

            images, labels = data[0].to(device), data[1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            mini_batch_loss = loss.item()

        # Test accuracy at the end of each epoch
        accuracy = test_model(model, testloader, device)
        log_output.append([epoch + 1, mini_batch_loss / len(trainloader),
                           accuracy])

        print("ep  {:03d}  loss  {:.3f}  acc  {:.3f}%".format(
            epoch + 1, mini_batch_loss / len(trainloader), accuracy))

        final_accuracy = accuracy

    if output_directory:
        if not retrain:
            file_path = os.path.join(output_directory, 'logger.csv')
            weight_path = os.path.join(output_directory, 'weights.pth')
        else:
            file_path = os.path.join(output_directory, 'logger_retrain.csv')
            weight_path = os.path.join(output_directory, 'weights_retrain.pth')

        # Save the training log
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(log_output)

        # Save the weights
        torch.save(model.state_dict(), weight_path)

    print("\nTraining complete.\n")

    return final_accuracy


def train_model_ignite(
        model, trainloader, testloader, optimizer,
        criterion, num_epochs, device,
        output_directory=None, retrain=False):
    """
    Function to train a model using the new ignite library.
    """
    metrics = {
        "Accuracy": Accuracy(),
        "Loss": Loss(criterion)
    }

    trainer = create_supervised_trainer(
        model, optimizer, criterion, device=device
    )
    evaluator = create_supervised_evaluator(
        model, metrics=metrics, device=device
    )

    log_output = []
    t0 = time.time()

    @trainer.on(Events.STARTED | Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(testloader)
        metrics = evaluator.state.metrics

        epoch_num = engine.state.epoch
        accuracy = metrics['Accuracy'] * 100
        loss = metrics['Loss']

        print(f"ep  {epoch_num:03d}  loss  {loss:.3f}  acc  {accuracy:.2f}%     avg time per epoch = {(time.time() - t0)/(epoch_num+1):.3f}s")
        
        log_output.append([epoch_num, loss, accuracy])

    @trainer.on(Events.COMPLETED)
    def save_state_dict_and_log(engine):
        if not output_directory:
            pass
        else:
            if not retrain:
                file_path = os.path.join(output_directory, 'logger.csv')
                weight_path = os.path.join(output_directory, 'weights.pth')
            else:
                file_path = os.path.join(output_directory,
                                         'logger_retrain.csv')
                weight_path = os.path.join(output_directory,
                                           'weights_retrain.pth')

            # Save the training log
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(log_output)

            # Save the weights
            torch.save(model.state_dict(), weight_path)

    trainer.run(trainloader, num_epochs)

    return log_output[-1][-1]
