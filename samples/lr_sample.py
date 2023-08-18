import matplotlib.pyplot as plt
import math

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    return lr

# Define the parameters
class Args:
    lr = 0.0003
    min_lr = 1e-4
    warmup_epochs = 5
    epochs = 130


def main():
    args = Args()

    # Create lists to store epoch and learning rate values
    epochs = list(range(args.epochs))
    learning_rates = []

    # Calculate and store learning rates for each epoch
    for epoch in epochs:
        lr = adjust_learning_rate(None, epoch, args)
        learning_rates.append(lr)

    # Plot the learning rate chart
    plt.plot(epochs, learning_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()



