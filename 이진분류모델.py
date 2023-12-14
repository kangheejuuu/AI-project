import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

if __name__ == '__main__':
    


    # 데이터 전처리 및 로드
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),
    }
    print(1)

    data_dir = 'C:/FireDisaster' # 경로 수정
    train_dir = 'train'
    valid_dir = 'valid'

    print(2)
    
    train_data_path = os.path.join(data_dir, train_dir)
    valid_data_path = os.path.join(data_dir, valid_dir)

    print(3)
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=0) for x in ['train', 'valid']}

    print(4)
    # 화재 감지 모델 정의
    class FireDetectionModel(nn.Module):
        def __init__(self):
            super(FireDetectionModel, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            
            )
            self.classifier = nn.Sequential(
                nn.Linear(64 * 112 * 112, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(256, 2),  # 클래스 개수에 맞게 조정 (fire과 non-fire이므로 2)
            )

        print(5)
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    model = FireDetectionModel()

    # 손실 함수와 최적화 기법 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 필요에 따라 조절
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)


    # 학습
    num_epochs = 30  # 필요에 따라 조정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(6)

    for epoch in range(num_epochs):
        print(f'{epoch + 1}/{num_epochs} epochs')
        print(f'Training...')
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

                # Gradient Clipping 추가
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            
             # 추가된 디버깅 출력
            print(f'Batch Loss: {loss.item():.4f}, Batch Acc: {torch.sum(preds == labels.data).double() / inputs.size(0):.4f}')

        epoch_loss = running_loss / len(image_datasets['train'])
        epoch_acc = running_corrects.double() / len(image_datasets['train'])
        print(f'Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

        print(f'Validation...')
        model.eval()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders['valid']:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets['valid'])
        epoch_acc = running_corrects.double() / len(image_datasets['valid'])
        print(f'Valid Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')



    # 학습된 모델 저장
    torch.save(model.state_dict(), 'fire_detection_model.pth')
