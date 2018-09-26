# mnist_LeNet5
使用近似LeNet5的结构实现对mnist手写数字的识别，大体的框架与mnist_MLP相似，不同的是改变神经网络结构，使用卷积层，池化层，全连接层代替之前简单的两层神经网络，需要注意的问题有以下几点：
- 卷积层的输入是4维的， [batch_size, image_size, image_size, num_channel]
- 卷积核即过滤器的大小需要设计为[filter2_size, filter2_size, layer1_deep,layer2_deep]， `layer1_deep`指的是上一层的输出深度，`layer2_deep`指的是当前层的输出维度，而`filter2_size`指的是应用于当前层的卷积核大小
- 需要注意stride，padding参数的设计
- 只有全连接层的权重需要加入正则化
- dropout一般只在全连接层而不是卷积层或者池化层使用</br>
`mnist_inference.py`设计了神经网络结构，定义了前向传播过程以及参数，`mnist_train.py`定义输入、输出以及loss的计算，并将训练的模型保存下来，方便之后的测试，`mnist_eval.py`用来计算验证集在模型上的准确率，经过上述训练过程，在验证集上的精度达到了99.3%。
