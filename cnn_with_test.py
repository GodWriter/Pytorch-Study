#coding=utf-8
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]
transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])

#训练集，将相对目录./data下的cifar-10-batches-py文件夹中的全部数据（50000张图片作为训练数据）加载到内存中，若download为True时，会自动从网上下载数据并解压
trainset = torchvision.datasets.CIFAR10(root='E:/Code_package/pythonwork/Pytorch/data', train=True, download=False, transform=transform)

#将训练集的50000张图片划分成12500份，每份4张图，用于mini-batch输入。shffule=True在表示不同批次的数据遍历时，打乱顺序。num_workers=2表示使用两个子进程来加载数据
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, 
                                          shuffle=True, num_workers=2)

#测试集，将相对目录./data下的cifar-10-batches-py文件夹中的全部数据（10000张图片作为测试数据）加载到内存中，若download为True时，会自动从网上下载数据并解压
testset = torchvision.datasets.CIFAR10(root='E:/Code_package/pythonwork/Pytorch/data', train=False, download=False, transform=transform)

#将测试集的10000张图片划分成2500份，每份4张图，用于mini-batch输入。
testloader = torch.utils.data.DataLoader(testset, batch_size=4, 
                                          shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 定义conv1函数的是图像卷积函数：输入为图像（3个频道，即彩色图）,输出为6张特征图, 卷积核为5x5正方形
        self.pool  = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

criterion = nn.CrossEntropyLoss() #叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  #使用SGD（随机梯度下降）优化，学习率为0.001，动量为0.9

for epoch in range(2): # 遍历数据集两次
    
    running_loss = 0.0
    #enumerate(sequence, [start=0])，i序号，data是数据
    for i, data in enumerate(trainloader, 0): 
        # get the inputs
        inputs, labels = data   #data的结构是：[4x3x32x32的张量,长度4的张量]
        
        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)  #把input数据从tensor转为variable
        
        # zero the parameter gradients
        optimizer.zero_grad() #将参数的grad值初始化为0
        
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels) #将output和labels使用叉熵计算损失
        loss.backward() #反向传播
        optimizer.step() #用SGD更新参数
        
        # 每2000批数据打印一次平均loss值
        running_loss += loss.data[0]  #loss本身为Variable类型，所以要使用data获取其Tensor，因为其为标量，所以取0
        if i % 2000 == 1999: # 每2000批打印一次
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    #print outputs.data
    _, predicted = torch.max(outputs.data, 1)  #outputs.data是一个4x10张量，将每一行的最大的那一列的值和序号各自组成一个一维张量返回，第一个是值的张量，第二个是序号的张量。
    total += labels.size(0)
    correct += (predicted == labels).sum()   #两个一维张量逐行对比，相同的行记为1，不同的行记为0，再利用sum(),求总和，得到相同的个数。

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


'''
假设共有50000条数据,首先我们初始化loss和优化函数
  >criterion = nn.CrossEntropLoss() #交叉熵损失函数
  >optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #使用SGD(随机梯度下降)优化，学习率为0，001，动量为0.9
                                                                   #optimizer能保持当前参数状态并基于计算得到的梯度进行更新
                                                                   #optimizer中被优化的参数必须是Variable,net.parameters是Variable的子类
卷积训练过程如下：
1.将50000条小数据分成batch=4的12500份数据，通俗说：将50000条数据，4个一组，分成12500组
2.现在设置要训练的次数，每次训练的数据是所有的12500组数据，每次训练详情如下：
  >获取一组数据（共4个）的数据和标签 --> input, labels = data #data的结构是:[4x3x32x32张量, 长度4的张量]
  >把获得的数据由Tensor转换为Variable --> inputs, labels = Variable(inputs), Variable(labels)
  >将optimizer中所有的参数置0，因为grad值是累加的，设置为0后，每次反向传播之后grad就会重新更新，而不是累加在上一次的grad的值上。结合上述步骤，
   每4组数据做一次前向传播，然后计算loss，之后进行反向传播，对所有的参数进行更新。更新公式即梯度下降公式，故每次grad都要在更新之前清0，否则计算
   出来的梯度会加上上次计算的梯度。 --> optimizers.zero_grad()
  >做好一切准备之后，进行前向传播，将数据集放到卷积神经网络的入口，数据开始进行前向传播。 -->outputs = net(inputs)
  >前向传播之后，将得到的结果与正确数据分类标签使用交叉熵计算损失 --> loss = criterion(outputs, labels)
  >进行反向传播，传播会让每一个网络参数的grad值进行更新，我们网络中每一个参数都是Variable类型，并且均是叶子节点，grad值必会更新 --> loss.backward()
  >使用SGD更新参数
 重复以上步骤，进行多次的训练，确保得到的loss最小
'''

