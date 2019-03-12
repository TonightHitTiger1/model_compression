import model_manager
import torch
import os
# import datasets
import cnn_models.conv_forward_model as convForwModel
import cnn_models.conv_forward_model1 as convForwModel1
import cnn_models.help_fun as cnn_hf
import quantization
import pickle
import copy
import quantization.help_functions as qhf
import functools
import helpers.functions as mhf
import sys
import torch.nn as nn
from flops_counter import get_model_complexity_info
from torchsummary import summary
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import time

datasets.BASE_DATA_FOLDER = '...'
SAVED_MODELS_FOLDER = 'Save_Model'
USE_CUDA = torch.cuda.is_available()

# print('CUDA_VISIBLE_DEVICES: {}'.format(os.environ['CUDA_VISIBLE_DEVICES'])


epochsToTrainCIFAR = 60
USE_BATCH_NORM = True
AFFINE_BATCH_NORM = True

TRAIN_TEACHER_MODEL = False
TRAIN_SMALLER_MODEL = False
TRAIN_SMALLER_QUANTIZED_MODEL = True
TRAIN_DISTILLED_MODEL = False
TRAIN_DIFFERENTIABLE_QUANTIZATION = False
CHECK_PM_QUANTIZATION = False

batch_size = 25

train_dataset = datasets.CIFAR10(root='.../CIFAR10', train=True, download=False,
                                 transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True)

test_dataset = datasets.CIFAR10(root='.../CIFAR10', train=False, download=False,
                                transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size, shuffle=True)

# Teacher model
model_name = 'TeacherModel'
teacherModel = convForwModel.ConvolForwardNet(**convForwModel.teacherModelSpec,
                                              useBatchNorm=USE_BATCH_NORM,
                                              useAffineTransformInBatchNorm=AFFINE_BATCH_NORM)

studentModel = convForwModel.ConvolForwardNet(**convForwModel.teacherModelSpec,
                                              useBatchNorm=USE_BATCH_NORM,
                                              useAffineTransformInBatchNorm=AFFINE_BATCH_NORM)

if USE_CUDA: teacherModel = teacherModel.cuda()
if USE_CUDA: studentModel = studentModel.cuda()
if not model_name in cifar10Manager.saved_models:
    cifar10Manager.add_new_model(model_name, teacherModelPath,
                                 arguments_creator_function={**convForwModel.teacherModelSpec,
                                                             'useBatchNorm': USE_BATCH_NORM,
                                                             'useAffineTransformInBatchNorm': AFFINE_BATCH_NORM})

# teacherModel.load_state_dict(torch.load("Save_Model/cifar10_pruning/cifar10_teacher.pkl"))
# studentModel.load_state_dict(torch.load("Save_Model/cifar10_pruning/cifar10_pruning19_retrain.pkl"))

tea_acc = cnn_hf.evaluateModel(teacherModel, test_loader, k=1)
# stu_acc = cnn_hf.evaluateModel(studentModel, test_loader, k=1)

print("teacher acc=",tea_acc)
# print("stu_pruning19_kd_acc=",stu_acc)

# Judging whether the filter has been clipped
# count = 0
# for p in studentModel.parameters():
#     if len(p.data.size()) == 4:
#         print("len of data",len(p.data))
#         print(p.data.size())
#         for i in range(len(p.data)):
#             print("a=",i)
#             a = (p.data[i].cpu() == torch.zeros(3, len(p.data), 3)).all().numpy()
#             if a:
#                count+=1
#     # break



# The training mode can be selected by adjusting the parameters in arguments_train_function.
# only train model: using 'epochs_to_train'
# quantization: using 'epochs_to_train'+'quantizeWeights'+'numBits'
# knowledge distillation: using 'epochs_to_train'+'use_distillation_loss'+'teacher_model'
# quantization+knowledge distillation: using all parameters

Numbits=[4,8]
for numBit in Numbits:
    arguments_train_function={'epochs_to_train': epochsToTrainCIFAR,
                              'quantizeWeights':True,
                              'numBits':numBit,
                              'use_distillation_loss': True,
                              'teacher_model':teacherModel,
                            }
    _, infoDict = convForwModel.train_model(teacherModel, train_loader=train_loader, test_loader=test_loader,
                                      **arguments_train_function)
    stu_acc = cnn_hf.evaluateModel(teacherModel, test_loader, k=1)
    print("quanti kd train  acc = ", stu_acc)
    torch.save(studentModel.state_dict(), "Save_Model/cifar10_new/cifar10_quanti_kd.pkl")



