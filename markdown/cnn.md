
Convolutional Neural Networks
---
> 白雪峰 --- xfbai@mtlab.hit.edu.cn.

##Outline 
- CNN栗子镇楼
- What is CNN
	- 什么是卷积
	- 什么是池化
- Why CNN 
- 对CNN的其他一些理解
- CNN实现（接口）


## 1. CNN栗子（A Beginning Glimpse of CNN）
(1)Modern CNN since Yann LeCun
![enter image description here](https://lh3.googleusercontent.com/IoYo64NskyHdxzltw-xEk5abHX1_Lmk8uTziFrd4S8C6fZdzdoMDd_x1FOXQiV9CQ4VzuC_xZNQ=s0 "lecun.png")
(2)
![DeepID](https://lh3.googleusercontent.com/i5NTRpYY7AGaDoLpnkKEtvVdCKYkEYswIBdlthKLjAyjp2uZtQ1f_pIqFXqI5dtxUAG4hT3Qq2E=s0 "DeepId.png")

## 2. What is CNN?
神经网络？卷积？
### 2.1 什么是卷积？
#### 卷积的定义: 

其连续的定义为：
$(f*g)(n) = \int_{ - \infty }^{ + \infty } {f(t)g(n-t)dt}$
其离散的定义为：
 $(f*g)(n) = \sum_{t = - \infty} ^{ + \infty } {f(t)g(n-t)}$
 
特点：　
![enter image description here](https://lh3.googleusercontent.com/kHgn5WtCxzUHBbh9GR-EoW0WB1NS5mbRzAGT4LpbwqrvACwDERgXIRaecJ4E4a9JiLavTA8GLKs=s0 "卷积1.jpg")

###2.2 离散卷积的栗子：
丢骰子，两个骰子加起来要等于4的概率是多少？
> $(f*g)(4) = \sum_{m = 1} ^{ 3 } {f(m)g(4-m)}$

####a. 二维离散的卷积：
$(f*g)(m,n) = \sum\limits_{k=0}^{2}\sum\limits_{h=0}^{2}f(h,k)g(m-h,n-k)$
则 (f*g)(1,1) ?
![enter image description here](https://lh3.googleusercontent.com/tbQsGAO_VeDVH4mtHlrY4F43alqwR2KrMbNfFuXOVEqmebEo_qmrwvPhNVZ4XCJOebz5Bv6Ujvk=s0 "111.gif")

> 离散的卷积可以看成是矩阵的乘法（翻转的）
###2.3 用到二维图像上：
####a.关于卷积中常用到的一些概念：
> In **Image Processing**
– Convolution is always named **filtering**, and there are many famous **filters/convolution kernels** that ***extract intuitive features in images***

![enter image description here](https://lh3.googleusercontent.com/OVoi8qCglOwEJ0OaJPT8cXZW10fj37-eGCSZLf81m33NF16pFg3kYsbz0rvAzvLYYer-9yD7Rks=s0 "222.gif")
通过在输入数据上不断移动卷积核，来提取不同位置的特征．

####b. 图像上作卷积的效果：
![enter image description here](https://lh3.googleusercontent.com/iCsuVDfVvDIRU5qvsUVfs14EV6ZL2y9joMy379_0HjfbWJEu1ieB0nwdv4vaysWSdiwZojlxUQc=s0 "a.png")
![enter image description here](https://lh3.googleusercontent.com/Cz0OC3027jUDRbR3M_MxUVEBgdP1hHXNkLZxQDrvfK6G55if9KB_of_Puqj5OCYm6zGmVgXJcnY=s0 "b.png")

###2.4 用到神经网络中
![enter image description here](https://lh3.googleusercontent.com/bOXd_J8N3dttwdyGUZpeoNkCX5GXHJdNe0QXdQfvGdad46Ue51n0LwE_vqtZLM1bqLtuvrV3aCs=s0 "c.png")

###2.5 卷积的细节
#### a. **filter/Kernel size, number**
假设神经网络的输入是６＊６的image,
![enter image description here](https://lh3.googleusercontent.com/d-JO3HlQ84n3J3ezH3yk7CxxdMkrZwwbMQaSWPFH2G0WyLxHbDXk5x4T7g6Zc1IhbA8w_jsI5CI=s0 "e.png")
那么，，，

再来个更形象的：
![enter image description here](https://lh3.googleusercontent.com/DSd7g_WeKsPbijUx3lQ4GUsvne5AmBrCV_bodA55Rvy2T4cY3S8pi2oSvNehjx7xIHy03aSsq9g=s0 "f.png")

#### **b. Stride**
> The step size you take the filter to sweep the
image

![enter image description here](https://lh3.googleusercontent.com/-5Yz8fp7qtAlITJQ0tvqi_Are9FaWZRI9JqkywY-0Uqh_zI9NAgM01UxRK1F9XN5EDuuu422ees=s0 "g.png")

#### **c. Zero-padding**
> 1. A way not to ignore pattern on border
> 2. New image is smaller than the original image

![enter image description here](https://lh3.googleusercontent.com/z9QvwT3qvYWooGsSquyQkVZ_-UzQg7vyMfcru1JE0V8wPkauMLglRMEVW_HbLiLDrO4zJlEF1QA=s0 "i.png")

#### **d.Channel**
![enter image description here](https://lh3.googleusercontent.com/qiKdcJQ2RW9ReMczWCjPoVe8xHZMOdBUes_9oJ9e5syUvVmLVdyZ5z_2dPEf4vMBufdmwTN2wlY=s0 "j.png")

### 2.6 池化(Pooling)
> Pooling layers **subsample** their input
	 1. Spatial pooling (also called subsampling or
downsampling) ***reduces the dimensionality*** of
each feature map
	2. Retains the ***most important information***

####1. Max pooling例子：
![enter image description here](https://lh3.googleusercontent.com/Xi65Dgab6okAbsBrEWptHgdDdVD8YVw4h8J8Xryvh7UTccLti6dbtPzBFVhuSSpSrfScI6EQ9EA=s0 "k.png")

> Pooling is **unsensitive to local translation.**(局部不变性)
– “if we translate the input by a small amount,
the values of most of the pooled outputs do
not change.”
![enter image description here](https://lh3.googleusercontent.com/-6QNd1M91W90/WivaZhBVFoI/AAAAAAAAAD0/oiIGDrhrzDcnxDsNYSdDeyTPd10lRxzzACLcBGAs/s0/l.png "l.png")

####2. Pooling的变种
Pooling should be designed to fit specific
applications.

+ Max pooling
+ Average pooling
+ Min pooling
+ l 2 -norm pooling
+ Dynamic k-pooling
+ Etc.

####3. Pooling的性质
+ Makes the input representation (feature dim) smaller and more manageable
+ Reduces number of parameters and computations in the network, therefore, controlling overfitting
+ Makes the network invariant to small transformations, distortions and translations in the input image
+ Help us arrive at almost scale invariant representation of our image

### 2.7 Flatten
![enter image description here](https://lh3.googleusercontent.com/-lE-sEVVpfMM/WivcIGABLII/AAAAAAAAAEM/eDZ0M_s83O0wYgA0tCLqUq2DeZtDNI7dACLcBGAs/s0/m.png "m.png")

### 2.8 Convolution v.s. Fully Connected
![enter image description here](https://lh3.googleusercontent.com/-rYDsoOFK74o/WivdNg46dsI/AAAAAAAAAEc/-jrkYV_J6Og3Enz3sz89-eiWSaOVfQrEQCLcBGAs/s0/n.png "n.png")

![enter image description here](https://lh3.googleusercontent.com/-QW2ntGUCi8Q/WivdgjYoy5I/AAAAAAAAAEk/Y8Wnp92JzDgKsJPbsX9UinR8s8qCaiboQCLcBGAs/s0/o.png "o.png")

![enter image description here](https://lh3.googleusercontent.com/-k5Z2j5gi4Ps/WivdryPzHgI/AAAAAAAAAEs/kmQJiYec-PQuWxOHr7RAU3hL9WrGPmDmACLcBGAs/s0/p.png "p.png")

### 2.9 The whole CNN
![enter image description here](https://lh3.googleusercontent.com/-OuDdpGbIEDc/WivfRPXR6yI/AAAAAAAAAFQ/LZH707Lw7RksqdOIR49EsLBLj9QwSh_DACLcBGAs/s0/q.png "q.png")

![enter image description here](https://lh3.googleusercontent.com/-7WUYZyLYfXs/WivjAytq6DI/AAAAAAAAAG8/affWGlCh_HEl687Vv7Pv9ltPdAoismfFgCLcBGAs/s0/v.png "v.png")


##3. Why CNN
+ Some patterns are much smaller than the whole image. 
![enter image description here](https://lh3.googleusercontent.com/-9FLQCqfyOWw/WivgK7WwTyI/AAAAAAAAAFk/KG60O6Gpkug_Kf4xfTKj0WKrvssdnq-ygCLcBGAs/s0/r.png "r.png")

+ The same patterns appear in different regions.
![enter image description here](https://lh3.googleusercontent.com/-3fabpqD56a8/WivgYAKQ6OI/AAAAAAAAAFs/VeU-hNdETlEE9k5kBdwmem6Amwpzgx6yACLcBGAs/s0/s.png "s.png")

+ Subsampling the pixels will not change the object
![enter image description here](https://lh3.googleusercontent.com/-ZijUIGdmZpU/WivgtVbBp0I/AAAAAAAAAGI/ujDZAv_WFTMGRnZq0d3RfDS-heND6LR9ACLcBGAs/s0/t.png "t.png")

##Soga~

![enter image description here](https://lh3.googleusercontent.com/-XI7wZtxeE4w/Wivg9clXzmI/AAAAAAAAAGQ/eREvQ78goWcyW808lc4PTEYCC9-vVN82ACLcBGAs/s0/u.png "u.png")

## 4. 对CNN的其他一些理解
### 4.1 关于接受域（receptive field）
称在底层中影响上层输出单元 $s$ 的单元集合为 $s$ 的接受域(receptive field).
![enter image description here](https://lh3.googleusercontent.com/-0qaKzwQMhG4/Wivkh2ngmbI/AAAAAAAAAHU/7JshtCSdplELaS6jhMLUlSwIqxJjcpRYACLcBGAs/s0/w.png "w.png")

> 处于卷积网络更深的层中的单元,它们的接受域要比处在浅层的单元的接受域更大。如果网络还包含类似步幅卷积或者池化之类的结构特征, 这种效应会加强。这意味着在卷积网络中尽管直接连接都是很稀疏的,但处在更深的层中的单元可以间接地连接到全部或者大部分输入图像。(表现性能)

![enter image description here](https://lh3.googleusercontent.com/-T73eltpbGdw/WivkluABl6I/AAAAAAAAAHc/3YDAQHlk730xZ5Wf5BhxoT_1EIs3eDZfgCLcBGAs/s0/x.png "x.png")

### 4.2 卷积与池化作为一种无限强的先验
首先，弱先验具有较高的熵值，因此自由性较强．强先验具有较低的熵值，这样的先验在决定参数最终取值时可以起着非常积极的作用．

把卷积网络类比成全连接网络，但对于网络的权重具有无限强的先验．a) 所有隐藏单元的权重是共享的．b) 除了一些连续的小单元的权重外，其他的权重都是0．c) 池化也是一个无限强的先验：每个单元都具有对少量平移的不变性．

**卷积和池化可能导致欠拟合！**　任何其他先验类似,卷积和池化只有当先验的假设合理且正确时才有用。如果一项任务依赖于保存精确的空间信息,那么在所有的特征上使用池化将会增大训练误差。

**根据实际需求选取先验**

## CNN实现

### 1. 反向传播
基本与FFNNs相同．
### 2. 共享权值的梯度问题
一个常见的做法：取梯度的平均值
### 3. CNN in Keras
![enter image description here](https://lh3.googleusercontent.com/-NsqxmGLDxRo/WivqzH18xTI/AAAAAAAAAIQ/i1NSOeWdOCky9UcVj9IMMBo7mXvV2zsowCLcBGAs/s0/y.png "y.png")

![enter image description here](https://lh3.googleusercontent.com/-bA24w3VUuZc/WivrI0rqKLI/AAAAAAAAAIY/59Xd9-yWMUEHINaf3SblbAsesjzwRd5qgCLcBGAs/s0/z.png "z.png")

### 4. CNN in Pytorch
#### a) Pytorch 相关接口
torch.nn.Conv2d：
![enter image description here](https://lh3.googleusercontent.com/-NlH7_HP0w_U/Wi0L5wRdbVI/AAAAAAAAAJI/Q3WMTiiNpD4yT_62FfYtytCs6znDBdthwCLcBGAs/s0/zz.png "zz.png")

torch.nn.functional.max_pool2d：
![enter image description here](https://lh3.googleusercontent.com/-b0O13ZAJhyk/Wi0PM31K32I/AAAAAAAAAJY/SIZS0X5h2I81J53JDyLIyIrNa9skZhUUACLcBGAs/s0/z1.png "z1.png")
#### b) LeNet in PyTorch.

    import torch.nn as nn
    import torch.nn.functional as F
	
	class LeNet(nn.Module):
		def __init__(self):
			super(LeNet, self).__init__()
			self.conv1 = nn.Conv2d(1, 6, 5)   #in_channels:1, out_channels:6, kernel_size:5   
			self.conv2 = nn.Conv2d(6, 16, 5)
		    self.fc1   = nn.Linear(16*5*5, 120)
		    self.fc2   = nn.Linear(120, 84)
		    self.fc3   = nn.Linear(84, 10)
		
		def forward(self, x):
			out = F.relu(self.conv1(x))
			out = F.max_pool2d(out, 2)
			out = F.relu(self.conv2(out))
			out = out.view(out.size(0), -1)
			out = F.relu(self.fc1(out))
			out = F.relu(self.fc2(out))
			out = F.softmax(self.fc3(out))
			return out