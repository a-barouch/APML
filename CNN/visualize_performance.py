import matplotlib.pyplot as plt

""" 
plot the loss as a function of number of iterations
"""


def plot_loss(epochs, loss):
    plt.plot(epochs, loss)
    plt.title('Training loss as function of epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


""" plot the accuracy (total) as a function of number of iterations"""


def plot_accuracy(epochs, accuracy):
    plt.plot(epochs, accuracy)
    plt.title('Training accuracy as function of epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()


""" plot the accuracy of each class as a function of number of iterations"""


def class_accuracy_plot(epochs, class_accuracy):
    car_acc = []
    truck_acc = []
    cat_acc = []
    for i in range(len(class_accuracy)):
        car_acc.append(class_accuracy[i][0])
        truck_acc.append(class_accuracy[i][1])
        cat_acc.append(class_accuracy[i][2])
    plt.plot(epochs, car_acc, 'g', label='car accuracy')
    plt.plot(epochs, truck_acc, 'b', label='truck accuracy')
    plt.plot(epochs, cat_acc, 'r', label='cat accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training accuracy of each class as function of epochs')
    plt.legend()
    plt.show()


""" plot the accuracy of each class as a function of learning rate"""


def plot_learning_rate_class(lr, class_accuracy):
    car_acc = []
    truck_acc = []
    cat_acc = []
    for i in range(len(class_accuracy)):
        car_acc.append(class_accuracy[i][0])
        truck_acc.append(class_accuracy[i][1])
        cat_acc.append(class_accuracy[i][2])

    plt.plot(lr, car_acc, 'g', label='car accuracy')
    plt.plot(lr, truck_acc, 'b', label='truck accuracy')
    plt.plot(lr, cat_acc, 'r', label='cat accuracy')
    plt.xlabel('Learning rates')
    plt.ylabel('Class Accuracies')
    plt.title('Training accuracy of each class as function of learning rate')
    plt.xscale('log')
    plt.xticks([0.1, 0.01, 0.001, 0.0001, 0.00001], [0.1, 0.01, 0.001, 0.0001, 0.00001])
    plt.legend()
    plt.show()


""" plot the accuracy as a function of batch size"""


def plot_batch_size(bs, accuracy):
    plt.plot(bs, accuracy)
    plt.title('Training Loss as function of batch size')
    plt.xlabel('Batch size')
    plt.ylabel('Loss')
    plt.xscale('log')
    plt.xticks([1, 2, 4, 8, 16, 32, 64, 128, 5625], [1, 2, 4, 8, 16, 32, 64, 128, 5625])
    plt.show()


""" plot the accuracy of each class as a function of batch size"""


def plot_batch_size_class(bs, class_accuracy):
    car_acc = []
    truck_acc = []
    cat_acc = []
    for i in range(len(class_accuracy)):
        car_acc.append(class_accuracy[i][0])
        truck_acc.append(class_accuracy[i][1])
        cat_acc.append(class_accuracy[i][2])

    plt.plot(bs, car_acc, 'g', label='car accuracy')
    plt.plot(bs, truck_acc, 'b', label='truck accuracy')
    plt.plot(bs, cat_acc, 'r', label='cat accuracy')
    plt.xlabel('Batch size')
    plt.ylabel('Loss')
    plt.title('Training loss of each class as function of batch size')
    plt.xticks([1, 2, 4, 8, 16, 32, 64, 128, 5625], [1, 2, 4, 8, 16, 32, 64, 128, 5625])
    plt.xscale('log')
    plt.legend()
    plt.show()


""" plot the loss as a function of number of epochs for each lr"""


def plot_loss_lr(epochs, loss):
    plt.plot(epochs, loss[0], 'g', label='lr = 0.1')
    plt.plot(epochs, loss[1], 'b', label='lr = 0.01')
    plt.plot(epochs, loss[2], 'r', label='lr = 0.001')
    plt.plot(epochs, loss[3], 'm', label='lr = 0.0001')
    plt.plot(epochs, loss[4], 'c', label='lr = 0.00001')
    plt.title('Training loss as function of epochs for each learning rate value')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.show()
