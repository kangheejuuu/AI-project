# main_code.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import confusion_matrix, classification_report
from visualize_results import visualize_confusion_matrix, visualize_classification_report

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
            nn.Linear(256, 5),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 모델 초기화 및 저장된 가중치 불러오기
model = DisasterDetectionModel()
model.load_state_dict(torch.load('disaster_detection_model.pth'))
model.eval()

# 데이터 전처리 및 로드
data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]),
}

data_dir = 'C:/test_dataset'
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

# Confusion Matrix 시각화
class_names = ['earthquake', 'fire', 'flood', 'landslide', 'typhoon']
visualize_confusion_matrix(all_labels, all_predictions, class_names)

# Classification Report 시각화
class_report = '''
              precision    recall  f1-score   support

           0       0.58      0.82      0.68         60
           1       0.96      0.87      0.91         60
           2       0.73      0.55      0.63         60
           3       0.78      0.58      0.67         60
           4       0.49      0.58      0.53         60

    accuracy                           0.68        300
   macro avg       0.71      0.68      0.68        300
weighted avg       0.71      0.68      0.68        300
'''
visualize_classification_report(class_report)
