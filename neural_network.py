import torch
import numpy as np

class NeuralNetwork(torch.nn.Module):
    def __init__(self, layer_size):
        super().__init__()
        self.n_layers = len(layer_size) - 1
        all_W = []
        all_b = []
        for i in range(self.n_layers):
            # Create the matrix W and the vector b for each layer
            input_size = layer_size[i]
            output_size = layer_size[i+1]

            std_dev = np.sqrt((input_size + output_size)/2)
            
            layer_W = torch.normal(
                0, std_dev, (output_size, input_size),
                requires_grad=True
            )
            layer_b = torch.zeros(
                (output_size, 1),
                requires_grad=True
            )
            
            all_W.append(layer_W)
            all_b.append(layer_b)
            
        # By using this function, calling self.parameters() will return
        # the W and b of all layers
        self.W = torch.nn.ParameterList(all_W)
        self.b = torch.nn.ParameterList(all_b)
            
    def compute(self, input):
        # For each layer, we apply the transformation x_(k+1) = tanh(A x_k + b)
        # We use x_0 = input and output = x_n, where n is the number of layers
        x = input
        for layer_index in range(self.n_layers):
            W = self.W[layer_index]
            b = self.b[layer_index]
            
            x = torch.tanh(W @ x + b)
            
        return x


if __name__ == "__main__":
    # This is just for testing
    nn = NeuralNetwork([1, 3, 3, 1])
    
    with torch.no_grad():
        output = nn.compute(torch.tensor([[5.0, 8.0, 9.0, 7.0, 1.4]], dtype=torch.float64))
    
    # output.backward()

    for parameter in nn.parameters():
        print(parameter)