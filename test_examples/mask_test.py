import torch
import numpy as np


def replace_random_coord(tensor):
    # Generate random indices to replace elements
    indices_to_replace = np.random.choice(total_elements, num_elements_to_replace, replace=False)

    # Replace the elements with -1 at the selected indices
    tensor_flat = tensor.flatten()
    tensor_flat[indices_to_replace] = -1

    # Reshape the modified array back to the original tensor shape
    tensor_modified = tensor_flat.reshape(tensor.shape)
    return tensor_modified


if __name__ == '__main__':
    # Define the set of elements
    elements_set = [2, 3, 4, 5]

    # Generate the random tensor
    tensor1 = np.random.choice(elements_set, size=(2, 50, 4))
    tensor2 = np.random.choice(elements_set, size=(2, 50, 4))

    # Get the total number of elements in the tensor
    total_elements = tensor1.size

    # Number of elements to replace with -1
    num_elements_to_replace = 80

    torch.set_printoptions(threshold=np.inf)

    tensor_modified1 = torch.tensor(replace_random_coord(tensor1))
    tensor_modified2 = torch.tensor(replace_random_coord(tensor2))

    mask1 = torch.all(tensor_modified1 != -1, dim=-1)
    mask2 = torch.all(tensor_modified2 != -1, dim=-1)
    mask = torch.logical_and(mask1, mask2)

    print(tensor_modified1)
    print(tensor_modified1[mask])

    #print(tensor_modified1)
    #print(tensor_modified2)