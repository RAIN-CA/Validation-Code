import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
import torch
from torch import nn
from torchvision import models
from transformers import BertModel

nltk.download('punkt')

# 加载预训练的Word2Vec模型
word2vec_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

def read_video_paths(file_path):
    with open(file_path, 'r') as file:
        video_paths = [line.strip() for line in file.readlines()]
    return video_paths

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word in word2vec_model.key_to_index]
    return tokens

def get_word_vectors(tokens, max_length=10):
    word_vectors = [word2vec_model[word] for word in tokens if word in word2vec_model]
    if len(word_vectors) > max_length:
        word_vectors = word_vectors[:max_length]
    else:
        word_vectors += [np.zeros(300)] * (max_length - len(word_vectors))
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(300)

class VideoFrameDataset(Dataset):
    def __init__(self, video_paths, labels_df, transform=None):
        self.video_paths = video_paths
        self.labels_df = labels_df
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        cap = cv2.VideoCapture(video_path)
        video_id = video_path.split('/')[-1].split('.')[0]
        frames_data = []

        for _, row in self.labels_df[self.labels_df['video_id'] == video_id].iterrows():
            cap.set(cv2.CAP_PROP_POS_FRAMES, row['start_frame'])
            ret, frame = cap.read()
            if ret:
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if self.transform:
                    frame = self.transform(frame)
                tokens = preprocess_text(row['narration'])
                narration_vector = get_word_vectors(tokens)
                frames_data.append((frame, narration_vector))

        cap.release()
        return frames_data

# 数据集设置
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

labels_df = pd.read_csv('EPIC_train_action_labels.csv')
train_video_paths = read_video_paths('train.txt')
test_video_paths = read_video_paths('test.txt')

train_dataset = VideoFrameDataset(train_video_paths, labels_df, transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)  # 批量大小调整为8

test_dataset = VideoFrameDataset(test_video_paths, labels_df, transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)  # 批量大小调整为8

class ImageCaptioningModel(nn.Module):
    def __init__(self):
        super().__init__()
        print("Initializing CNN...")
        print("Loading ResNet18 weights...")
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        print("ResNet18 weights loaded successfully.")

        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        print("Initializing Transformer...")
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=2048, nhead=8), num_layers=6)
        print("Initializing GRU...")
        self.seq2seq = nn.GRU(2048, 300, batch_first=True)
        print("Initializing fully connected layer...")
        self.fc = nn.Linear(300, 300)


    def forward(self, images):
        image_features = self.cnn(images).squeeze(-1).squeeze(-1)
        encoded_features = self.transformer_encoder(image_features.unsqueeze(0))
        output, _ = self.seq2seq(encoded_features)
        final_output = self.fc(output.squeeze(0))
        return final_output

device = torch.device("cpu")  # 强制使用CPU
print("Creating model...")
model = ImageCaptioningModel().to(device)
print("Model created successfully.")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
print("q3")
loss_function = nn.MSELoss()
print("q4")

# 训练循环
for epoch in range(10):  # 假设训练10个epoch
    model.train()
    for frames, narration_vectors in train_loader:
        frames = frames.to(device)
        narration_vectors = narration_vectors.to(device)

        optimizer.zero_grad()
        outputs = model(frames)
        loss = loss_function(outputs, narration_vectors)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

def evaluate_model(model, test_loader, loss_function):
    model.eval()  # 设置模型为评估模式
    total_loss = 0
    with torch.no_grad():  # 在评估过程中不计算梯度
        for frames, narration_vectors in test_loader:
            frames = frames.to(device)
            narration_vectors = narration_vectors.to(device)
            
            outputs = model(frames)
            loss = loss_function(outputs, narration_vectors)
            total_loss += loss.item()
    
    average_loss = total_loss / len(test_loader)
    return average_loss

# 测试模型
test_loss = evaluate_model(model, test_loader, loss_function)
print(f'Test Loss: {test_loss:.4f}')
