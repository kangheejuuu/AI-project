# confusion matrix, classification report 출력하는 코드

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import confusion_matrix, classification_report

# 모델 정의
class DisasterDetectionModel(nn.Module):
    def __init__(self):
        super(DisasterDetectionModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
           
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 112 * 112, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 6),  # non_disaster 추가해서 6개로 변경
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 모델 초기화 및 저장된 가중치 불러오기
model = DisasterDetectionModel()
model.load_state_dict(torch.load('Disastermodel.pth'), strict=False)
model.eval()

# 데이터 전처리 및 로드
data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]),
}

data_dir = 'C:/1212_test'
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)                                                                             

# 예측 및 평가
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

# Confusion Matrix 출력
conf_matrix = confusion_matrix(all_labels, all_predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report 출력
class_report = classification_report(all_labels, all_predictions)
print("Classification Report:")
print(class_report)
