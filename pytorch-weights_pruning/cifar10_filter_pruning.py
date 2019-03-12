
import sys
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchsummary import summary
from thop import profile

from pruning.methods import filter_prune
from pruning.utils import to_var, train, test, prune_rate
from models import TeacherNet

# Hyper Parameters
param = {
    'pruning_perc': 11,
    'batch_size': 25,
    'test_batch_size':25,
    'num_epochs': 10,
    'learning_rate': 0.001,
    'weight_decay': 5e-4,
}

# Data loaders
train_dataset = datasets.CIFAR10(root='../data/Cifar10',train=True, download=True,
    transform=transforms.ToTensor())
loader_train = torch.utils.data.DataLoader(train_dataset,
    batch_size=param['batch_size'], shuffle=True)

test_dataset = datasets.CIFAR10(root='../data/Cifar10', train=False, download=True,
    transform=transforms.ToTensor())
loader_test = torch.utils.data.DataLoader(test_dataset,
    batch_size=param['test_batch_size'], shuffle=True)

# (76,3,3)  76:numFilters  (3,3):kernel sizes
teacherModelSpec = {'spec_conv_layers': [(76, 3, 3), (76, 3, 3), (126, 3, 3), (126, 3, 3), (148, 3, 3),
                                        (148, 3, 3), (148, 3, 3), (148, 3, 3)],
                    'spec_max_pooling': [(1,2,2), (3, 2, 2), (7, 2, 2)],
                    'spec_dropout_rates': [(1, 0.2), (3, 0.3), (7, 0.35), (8, 0.4), (9, 0.4)],
                    'spec_linear': [1200, 1200], 'width': 32, 'height': 32}


teacherNet = TeacherNet(**teacherModelSpec,
                        useBatchNorm=True,
                        useAffineTransformInBatchNorm=True)


teacherNet.load_state_dict(torch.load('/home/gpu/yang/Compression/pytorch-weights_pruning-master/pytorch-weights_pruning-master/models/cifar10_teacher.pkl'))
if torch.cuda.is_available():
    print('CUDA ensabled.')
    teacherNet.cuda()

print("--- Pretrained network loaded ---")

# summary(teacherNet,input_size=(3,32,32))

test(teacherNet, loader_test)


# for p in teacherNet.parameters():
#     if len(p.data.size()) == 4:
#         print("p", p)


optimizer = torch.optim.SGD(teacherNet.parameters(), lr=0.001, nesterov=True,
                          momentum=0.9, weight_decay=0.00022)
criterion = nn.CrossEntropyLoss().cuda()
train(teacherNet, criterion, optimizer, param, loader_train)

# torch.save(teacherNet.state_dict(),'/home/gpu/yang/Compression/pytorch-weights_pruning-master/pytorch-weights_pruning-master/models/cifar10_teacher_retrain.pkl')

test(teacherNet, loader_test)


# 开始剪枝
masks = filter_prune(teacherNet, param['pruning_perc'])
teacherNet.set_masks(masks)
print("--- {}% parameters pruned ---".format(param['pruning_perc']))
test(teacherNet, loader_test)


# 剪枝结束后 正确率急速下降
# 重训练
optimizer = torch.optim.SGD(teacherNet.parameters(), lr=0.001, nesterov=True,
                          momentum=0.9, weight_decay=0.00022)
criterion = nn.CrossEntropyLoss().cuda()
train(teacherNet, criterion, optimizer, param, loader_train)

torch.save(teacherNet.state_dict(),'/home/gpu/yang/Compression/pytorch-weights_pruning-master/pytorch-weights_pruning-master/models/cifar10_prune11.pkl')


# 再次测试结果
print("--- After retraining ---")
test(teacherNet, loader_test)
prune_rate(teacherNet)

