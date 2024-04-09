import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# 检查可用的GPU数量
device_count = torch.cuda.device_count()
if device_count == 0:
    raise RuntimeError("No GPUs available on this machine.")

device_id = [0, 1]
# 定义模型
model = models.resnet50()

# 将模型加载到多个GPU上
model = nn.DataParallel(model, device_ids=device_id)
model.cuda()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 模拟输入数据
input_data = torch.randn(1024, 3, 224, 224).cuda()

# 后台持续运行
while True:
    # 将数据输入模型并计算输出
    output = model(input_data)

    # 计算损失
    target = torch.randint(0, 100, (1024,)).cuda()
    loss = criterion(output, target)

    # 执行反向传播和优化步骤
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()