device = 'cuda'
import torch
import torchvision
import torchvision.transforms as tr
from torch.utils.data import DataLoader,Dataset
import numpy as np
import torch.optim as optim
from Model import Model
import torch.nn as nn
import torch.nn.functional as F

#데이터 전처리
transform = tr.Compose(
    [   tr.Resize([256,256]),
        tr.RandomRotation(90, expand=False),
        tr.ToTensor(),
     tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#데이터도 가져오고 lable까지 자동으로 매겨줌 + transf 로 정의해둔 데이터 전처리 까지 넣어서 데이터 전처리까지 가능함
trainset = torchvision.datasets.ImageFolder(root='manyimages2',transform=transform)


trainloader = DataLoader(trainset,batch_size = 64,shuffle = True)

model = Model([2,2,2,2]).to(device)

#손실함수 + Optimizer 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.005,betas=(0.9,0.999),eps=1e-08,weight_decay=0,amsgrad=False)


for epoch in range(100):
    
    running_loss = 0.0
    for batch_idx, (inputs,targets) in enumerate(trainloader, 0):
        # 입력을 받은 후,
        inputs, targets = inputs.to(device),targets.to(device)

        

        # 변화도 매개변수를 0으로 만든 후
        optimizer.zero_grad()

        # 학습 + 역전파 + 최적화
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        # 통계 출력
        running_loss += loss
        if batch_idx % 2 == 0:    
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, batch_idx + 1, running_loss))
            running_loss = 0.0

torch.save(model,'ManyImageV5Rotation.pt')

print('Finished Training')

