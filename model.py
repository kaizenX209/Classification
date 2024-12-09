import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50

# Khởi tạo mô hình ResNet50 và thay đổi số lớp đầu ra
model = resnet50()
num_classes = 3  # Số lớp trong mô hình đã train
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Tải state dict và map về CPU
state_dict = torch.load('skin_type_resnet50.pth', map_location=torch.device('cpu'), weights_only=True)
model.load_state_dict(state_dict)
model.eval()

def classify_image(image_path):
    # Đọnh nghĩa transform chuẩn hóa
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Đọc và tiền xử lý ảnh
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển từ BGR sang RGB
    img = cv2.resize(img, (224, 224))  # Resize ảnh
    
    # Chuyển đổi sang tensor và chuẩn hóa
    img = transform(img).unsqueeze(0)  # Thêm batch dimension
    
    # Dự đoán
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(probabilities, 1)
    
    # Phân loại với độ tin cậy
    prob_value = probabilities[0][predicted[0]].item()
    class_names = ["Dry", "Normal", "Oily"]
    result = class_names[predicted[0]]
    
    return result, prob_value

# Ví dụ sử dụng
result, confidence = classify_image('test/dry/dry_1d3b1b41c06745c89fb6_jpg.rf.9fe7b2181a36c2b6f67744be1af2fdd7.jpg')
print(f"Kết quả: {result}")
print(f"Độ tin cậy: {confidence:.2%}")
