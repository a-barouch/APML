import dataset
import models
from torch.utils.data import WeightedRandomSampler
import torch.optim as optim
import torch.nn as nn
import torch.utils.data.dataloader
import torchvision.transforms as transforms
import numpy as np

import visualize_performance
import advarsarial

PATH_TO_SAVE = './trained models/trained_model_arielle.ckpt'
# PATH_TO_LOAD = "data/pre_trained.ckpt" # given model
PATH_TO_LOAD = "./trained_model_arielle.ckpt"
PATH_TRAIN_FIXED = './data/train_fixed_v4.pickle'
PATH_DEV_FIXED = './data/dev_fixed_v2.pickle'
PATH_TRAIN_ORIGINAL = './data/train.pickle'
PATH_DEV_ORIGINAL = './data/dev.pickle'
NUM_OF_CLASSES = 3

classes = ('car', 'truck', 'cat')

idx2class = {0: 'car', 1: 'truck', 2: 'cat'}
class2idx = {'car': 0, 'truck': 1, 'cat': 2}


def get_class_distribution(dataset_obj):
    count_dict = {'car': 0, 'truck': 0, 'cat': 0}
    for element in dataset_obj:
        y_lbl = element[1]
        y_lbl = idx2class[y_lbl]
        count_dict[y_lbl] += 1

    return count_dict


def prepare_datasets(batch_size):
    # get data sets for train and test
    trainset = dataset.get_dataset_as_torch_dataset(PATH_TRAIN_FIXED)
    testset = dataset.get_dataset_as_torch_dataset(PATH_DEV_FIXED)

    # # define a weighted sampler
    # ws = weighted_sampler(trainset)

    # transform data
    transform = transforms.Compose(
        [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #    ,
    # transforms.RandomGrayscale(p=.25),
    # transforms.RandomAffine(20)])
    trainset.transform = transform
    testset.transform = transform

    # create train and test loaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)  # , sampler=ws)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
    return trainloader, testloader


"""
    calculate samples distribution to classes and arranges batches to originate from balanced distribution
"""


def weighted_sampler(trainset):
    # taken from: https://towardsdatascience.com/pytorch-basics-sampling-samplers-2a0f29f0bf2a

    class_count = [i for i in get_class_distribution(trainset).values()]
    class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
    trainloader1 = torch.utils.data.DataLoader(trainset, batch_size=5625,
                                               shuffle=True, num_workers=0)
    for data in trainloader1:
        images, labels = data
        class_weights_all = class_weights[labels]
    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True
    )
    return weighted_sampler


"""
    loops and updates weights 
"""


def training_loop(trainloader, lr, epochs):
    # weighted loss
    class_count = [i for i in get_class_distribution(trainloader.dataset).values()]
    class_weights = torch.FloatTensor([1 - (x / sum(class_count)) for x in class_count])

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    epoch_losses = []
    epoch_accuracies = []
    epoch_classes_accuracies = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        print("Epoch #" + str(epoch))
        running_loss = []
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss.append(loss.item())

        cur_loss = sum(running_loss) / len(running_loss)
        epoch_losses.append(cur_loss)
        epoch_accuracies.append(measure_accuracy())
        epoch_classes_accuracies.append(measure_classes_accuracy(batch_size))
        print('[%d] loss: %.3f' % (epoch + 1, sum(running_loss) / len(running_loss)))

    print('Finished Training')

    # save model
    model.save(PATH_TO_SAVE)
    return epoch_losses


"""
    measures overall accuracy of the model on the test set
"""


def measure_accuracy():
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    try:
        print('Accuracy of the network on the test images: %d %%' % (
                100 * correct / total))
        return 100 * correct / total
    except:
        return 1


"""
    measures accuracy of the model on the test set for each class
"""


def measure_classes_accuracy(batch_size):
    class_correct = list(0. for i in range(NUM_OF_CLASSES))
    class_total = list(0. for i in range(NUM_OF_CLASSES))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(batch_size):
                try:
                    label = labels[i]
                    if batch_size == 1:
                        class_correct[label] += c.item()
                        class_total[label] += 1
                    else:
                        class_correct[label] += c[i].item()
                        class_total[label] += 1

                except:
                    continue
    results = []
    for i in range(NUM_OF_CLASSES):
        try:
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))
            results.append(100 * class_correct[i] / class_total[i])
        except:
            continue
    print("Average accuracy: " + str(sum(results) / len(results)))
    return results


if __name__ == '__main__':

    # lr_list = [0.1,0.01,0.001,0.0001, 0.00001]
    # batch_size_list = [1, 2, 4, 8, 16, 32, 64, 128, 5625]

    # hyperparameters
    batch_size = 16
    lr = 0.001
    epochs = 50

    accuracies_list = []
    class_accuracies_list = []
    epoch_losses_list = []

    # train a model from training set
    to_train = True

    trainloader, testloader = prepare_datasets(batch_size)
    model = models.SimpleModel()

    # use given trained model
    if not to_train:
        model.load(PATH_TO_LOAD)
    else:
        epoch_losses_list.append(training_loop(trainloader, lr, epochs))

    # accuracy on all classes
    accuracies_list.append(measure_accuracy())

    print("Test set performance: \n")

    # accuracy per class
    class_accuracies_list.append(measure_classes_accuracy(batch_size))
