"""
AI 识鸟 App - 安卓版
摄像头检测到鸟就闪烁
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
import cv2
import torch
import timm
import numpy as np
from PIL import Image as PILImage
from torchvision import transforms

# 设置
DETECT_EVERY = 10
BIRD_RANGE = range(7, 25)
FLASH_COLOR = (255, 0, 0)  # 红色


class BirdDetectorApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        
        # 标题
        self.title_label = Label(
            text='AI 识鸟',
            font_size='24sp',
            size_hint=(1, 0.1)
        )
        self.layout.add_widget(self.title_label)
        
        # 摄像头画面
        self.image = Image(size_hint=(1, 0.8))
        self.layout.add_widget(self.image)
        
        # 状态文字
        self.status_label = Label(
            text='正在加载模型...',
            font_size='18sp',
            size_hint=(1, 0.1)
        )
        self.layout.add_widget(self.status_label)
        
        # 初始化变量
        self.model_loaded = False
        self.cap = None
        self.frame_num = 0
        self.bird_hits = 0
        self.flashing = False
        
        # 延迟加载模型（避免卡界面）
        Clock.schedule_once(self.load_model, 0)
        
        return self.layout
    
    def load_model(self, dt):
        try:
            self.status_label.text = '正在加载AI模型...'
            self.model = timm.create_model("efficientnet_b0", pretrained=True)
            self.model.eval()
            
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            
            self.model_loaded = True
            self.status_label.text = '模型加载完成！正在打开摄像头...'
            
            # 打开摄像头
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.status_label.text = '检测中... 对准鸟类试试！'
                Clock.schedule_interval(self.update_frame, 0.1)
            else:
                self.status_label.text = '无法打开摄像头！'
        except Exception as e:
            self.status_label.text = f'加载失败: {str(e)}'
    
    def is_bird(self, frame):
        img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensor = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            pred = self.model(tensor).argmax(dim=1).item()
        return pred in BIRD_RANGE
    
    def update_frame(self, dt):
        if not self.cap or not self.cap.isOpened():
            return
        
        ok, frame = self.cap.read()
        if not ok:
            return
        
        self.frame_num += 1
        
        # 检测鸟
        if self.frame_num % DETECT_EVERY == 0:
            if self.is_bird(frame):
                self.bird_hits += 1
            else:
                self.bird_hits = 0
            self.flashing = (self.bird_hits >= 3)
        
        # 显示画面
        if self.flashing:
            frame = np.full_like(frame, FLASH_COLOR)
            cv2.putText(frame, "BIRD!", (80, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
            self.status_label.text = '🐦 检测到鸟！'
            self.status_label.color = (1, 0, 0, 1)
        else:
            self.status_label.text = '检测中...'
            self.status_label.color = (0, 1, 0, 1)
        
        # 转换为 Kivy 纹理
        buf = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture
    
    def on_stop(self):
        if self.cap:
            self.cap.release()


if __name__ == '__main__':
    BirdDetectorApp().run()
