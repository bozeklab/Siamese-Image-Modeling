import matplotlib.pyplot as plt
import math
import torch

class Args:
    lr = 0.0003
    min_lr = 1e-4
    warmup_epochs = 10
    epochs = 50

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    return lr

def exponential_lr_schedule(epoch, args):
    base_lr = args.lr
    gamma = 0.85
    lr = base_lr * math.pow(gamma, epoch / args.epochs)
    return lr

def main():
    args = Args()

    # Create lists to store epoch and learning rate values
    epochs = list(range(args.epochs))
    learning_rates_adjusted = []
    learning_rates_exponential = []

    # Calculate and store learning rates for each epoch using adjusted method
    for epoch in epochs:
        lr_adjusted = adjust_learning_rate(None, epoch, args)
        learning_rates_adjusted.append(lr_adjusted)

    # Calculate learning rates for each epoch using exponential schedule
    for epoch in epochs:
        lr_exponential = exponential_lr_schedule(epoch, args)
        learning_rates_exponential.append(lr_exponential)

    # Plot the learning rate chart
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, learning_rates_adjusted, label='Adjusted LR Schedule')
    plt.plot(epochs, learning_rates_exponential, label='Exponential LR Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedules')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()