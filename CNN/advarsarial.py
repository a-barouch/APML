from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

import dataset

PATH_TO_LOAD = "data/for_adversarial.ckpt"
PATH_DEV_FIXED = './data/dev_fixed_v2.pickle'


def generate_adversarial(model):
    model.load(PATH_TO_LOAD)
    test_set = dataset.get_dataset_as_torch_dataset(PATH_DEV_FIXED)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                              shuffle=True, num_workers=0)
    # adversarial attack
    accuracies = []
    examples = []
    # Run test for each epsilon
    epsilons = [0, .05, .1, .15, .2, .25, .3]
    for eps in epsilons:
        acc, ex = create_perturbed_image(model, test_loader, eps)
        accuracies.append(acc)
        examples.append(ex)
    plot_example(epsilons, examples)


# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def create_perturbed_image(model, testloader, epsilon):
    correct = 0
    adv_examples = []
    model.eval()
    for data, target in testloader:

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue
        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output2 = model(perturbed_data)

        # Check for success
        final_pred = output2.max(1, keepdim=True)[1]  # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze()
                adv_examples.append((init_pred, final_pred, adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze()
                adv_examples.append((init_pred, final_pred, adv_ex))

        # Calculate final accuracy for this epsilon
        final_acc = correct / float(len(testloader))
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(testloader), final_acc))

        # Return the accuracy and an adversarial example
        return final_acc, adv_examples


def plot_example(epsilons, examples):
    # Plot several examples of adversarial samples at each epsilon
    cnt = 0
    plt.figure(figsize=(8, 10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons), len(examples[0]), cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig, adv, ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(dataset.un_normalize_image(ex.detach()), cmap="gray")
            # plt.imshow(ex, cmap="gray")

    plt.tight_layout()
    plt.show()
