import torch
import torch.nn as nn
from pruning.layers import MaskedLinear, MaskedConv2d 
from torch.nn.init import xavier_uniform, calculate_gain
import torch.nn.functional as Functional

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = MaskedLinear(28*28, 200)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MaskedLinear(200, 200)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear3 = MaskedLinear(200, 10)
        
    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.relu1(self.linear1(out))
        out = self.relu2(self.linear2(out))
        out = self.linear3(out)
        return out

    def set_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future
        self.linear1.set_mask(masks[0])
        self.linear2.set_mask(masks[1])
        self.linear3.set_mask(masks[2])
    

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = MaskedConv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2 = MaskedConv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv3 = MaskedConv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.linear1 = nn.Linear(7*7*64, 10)
        
    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = self.relu3(self.conv3(out))
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        return out

    def set_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future
        self.conv1.set_mask(torch.from_numpy(masks[0]))
        self.conv2.set_mask(torch.from_numpy(masks[1]))
        self.conv3.set_mask(torch.from_numpy(masks[2]))


class TeacherNet(nn.Module):

    ''' Teacher model as described in the paper :
    "Do deep convolutional neural network really need to be deep and convolutional?"'''

    def __init__(self, width, height, spec_conv_layers, spec_max_pooling, spec_linear, spec_dropout_rates, useBatchNorm=False,
                 useAffineTransformInBatchNorm=False):
        '''
        The structure of the network is: a number of convolutional layers, intermittend max-pooling and dropout layers,
        and a number of linear layers. The max-pooling layers are inserted in the positions specified, as do the dropout
        layers.

        :param spec_conv_layers: list of tuples with (numFilters, width, height) (one tuple for each layer);
        :param spec_max_pooling: list of tuples with (posToInsert, width, height) of max-pooling layers
        :param spec_dropout_rates list of tuples with (posToInsert, rate of dropout) (applied after max-pooling)
        :param spec_linear: list with numNeurons for each layer (i.e. [100, 200, 300] creates 3 layers)
        '''

        super(TeacherNet, self).__init__()

        self.width = width
        self.height = height
        self.conv_layers = []
        self.max_pooling_layers = []
        self.dropout_layers = []
        self.linear_layers = []
        self.max_pooling_positions = []
        self.dropout_positions = []
        self.useBatchNorm = useBatchNorm
        self.batchNormalizationLayers = []

        #creating the convolutional layers
        oldNumChannels = 3
        for idx in range(len(spec_conv_layers)):
            currSpecLayer = spec_conv_layers[idx]
            numFilters = currSpecLayer[0]
            kernel_size = (currSpecLayer[1], currSpecLayer[2])
            #The padding needs to be such that width and height of the image are unchanges after each conv layer
            padding = ((kernel_size[0]-1)//2, (kernel_size[1]-1)//2)
            newConvLayer = MaskedConv2d(in_channels=oldNumChannels, out_channels=numFilters,
                                                                    kernel_size=kernel_size, padding=padding)
            xavier_uniform(newConvLayer.weight, calculate_gain('conv2d')) #glorot weight initialization
            self.conv_layers.append(newConvLayer)
            self.batchNormalizationLayers.append(nn.BatchNorm2d(numFilters,
                                                            affine=useAffineTransformInBatchNorm))
            oldNumChannels = numFilters

        #creating the max pooling layers
        for idx in range(len(spec_max_pooling)):
            currSpecLayer = spec_max_pooling[idx]
            kernel_size = (currSpecLayer[1], currSpecLayer[2])
            self.max_pooling_layers.append(nn.MaxPool2d(kernel_size))
            self.max_pooling_positions.append(currSpecLayer[0])

        #creating the dropout layers
        for idx in range(len(spec_dropout_rates)):
            currSpecLayer = spec_dropout_rates[idx]
            rate = currSpecLayer[1]
            currPosition = currSpecLayer[0]
            if currPosition < len(self.conv_layers):
                #we use dropout2d only for the conv_layers, otherwise we use the usual dropout
                self.dropout_layers.append(nn.Dropout2d(rate))
            else:
                self.dropout_layers.append(nn.Dropout(rate))
            self.dropout_positions.append(currPosition)


        #creating the linear layers
        oldInputFeatures = oldNumChannels * width * height // 2**(2*len(self.max_pooling_layers))
        for idx in range(len(spec_linear)):
            currNumFeatures = spec_linear[idx]
            newLinearLayer = nn.Linear(in_features=oldInputFeatures, out_features=currNumFeatures)
            xavier_uniform(newLinearLayer.weight, calculate_gain('linear'))  # glorot weight initialization
            self.linear_layers.append(newLinearLayer)
            self.batchNormalizationLayers.append(nn.BatchNorm1d(currNumFeatures,
                                                                                 affine=useAffineTransformInBatchNorm))
            oldInputFeatures = currNumFeatures

        #final output layer
        self.out_layer = nn.Linear(in_features=oldInputFeatures, out_features=10)
        xavier_uniform(self.out_layer.weight, calculate_gain('linear'))


        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.max_pooling_layers = nn.ModuleList(self.max_pooling_layers)
        self.dropout_layers = nn.ModuleList(self.dropout_layers)
        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.batchNormalizationLayers = nn.ModuleList(self.batchNormalizationLayers)
        self.num_conv_layers = len(self.conv_layers)
        self.total_num_layers = self.num_conv_layers + len(self.linear_layers)

    def forward(self, input):

        for idx in range(self.total_num_layers):
            if idx < self.num_conv_layers:
                input = Functional.relu(self.conv_layers[idx](input))
            else:
                if idx == self.num_conv_layers:
                    #if it is the first layer after the convolutional layers, make it as a vector
                    input = input.view(input.size()[0], -1)
                input = Functional.relu(self.linear_layers[idx-self.num_conv_layers](input))

            if self.useBatchNorm:
                input = self.batchNormalizationLayers[idx](input)

            try:
                posMaxLayer = self.max_pooling_positions.index(idx)
                input = self.max_pooling_layers[posMaxLayer](input)
            except ValueError: pass

            try:
                posDropoutLayer = self.dropout_positions.index(idx)
                input = self.dropout_layers[posDropoutLayer](input)
            except ValueError: pass

        input = Functional.relu(self.out_layer(input))

        #No need to take softmax if the loss function is cross entropy
        return input

    def set_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future
        for i,l in enumerate(self.conv_layers):
            l.set_mask(torch.from_numpy(masks[i]))
