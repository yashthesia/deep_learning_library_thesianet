import numpy as np

class Convo:
    def __init__(self,
                 input_size,
                 filter_size=3,
                 number_of_filters=64,
                 padding=0,
                 stride=1):
        self.input_size = input_size  # (H, W, D)
        self.batch, self.H_in, self.W_in, self.D_in = self.input_size
        self.D_out = number_of_filters
        self.filer_size = filter_size
        self.padding = padding
        self.stride = stride

        self.params = {}
        self.grad = {}
        self.params['w'] = np.random.randn(self.filer_size, self.filer_size, self.D_in, self.D_out)
        self.params['b'] = np.random.randn(self.D_out)

    def forward(self, input):

        # save the input for gradient calculation and add padding on it
        self.input = np.pad(input, [(0, 0), (self.padding,self.padding),(self.padding,self.padding),(0,0)], mode='constant', constant_values=0)
        # print(self.input.shape)
        #calculating output dimensions
        self.batch, _, _, _ = input.shape
        self.H_out = ((self.H_in + 2*self.padding -self.filer_size)//(self.stride)) +1
        self.W_out = ((self.W_in + 2 * self.padding - self.filer_size) // (self.stride)) + 1

        out = np.random.randn(self.batch, self.H_out, self.W_out, self.D_out)*0

        for m in range(self.batch):
           for i in range(0,self.input.shape[1]-self.filer_size+1,self.stride):
               for j in range(0,self.input.shape[2]-self.filer_size+1,self.stride):
                   for f in range(self.D_out):
                       out[m,i,j,f] = np.sum(np.multiply(self.input[m,i:i+self.filer_size,j:j+self.filer_size,:],self.params['w'][:,:,:,f])) + self.params['b'][f]

        return out

    def backward(self, grad):

        self.grad['w'] = np.random.randn(self.filer_size, self.filer_size, self.D_in, self.D_out)*0
        self.grad['b'] = np.random.randn(self.D_out)*0
        self.grad['input'] = np.random.randn(self.input.shape[0], self.input.shape[1], self.input.shape[2], self.input.shape[3]) * 0


        for m in range(self.batch):
            for i in range(0, self.input.shape[1] - self.filer_size + 1, self.stride):
                for j in range(0, self.input.shape[2] - self.filer_size + 1, self.stride):
                    for f in range(self.D_out):
                        self.grad['input'][m,i:i+self.filer_size,j:j+self.filer_size,:] += self.params['w'][:,:,:,f]*grad[m,i,j,f]
                        self.grad['w'][:,:,:,f] += self.input[m,i:i+self.filer_size,j:j+self.filer_size,:]*grad[m,i,j,f]
                        self.grad['b'][f] = grad[m,i,j,f]

        self.grad['input'] = self.grad['input'][:,self.padding:-self.padding,self.padding:-self.padding]
        return grad

