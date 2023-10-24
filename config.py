import yaml

class Config:
    def __init__(self):
        self.video = 'rtsp://192.168.1.50/channel_1'
        self.monitor = False
        self.yolo = ['person', 'cat', 'dog']
        self.model = 'yolov8n'
        self.threshold = 350
        self.start_frames = 3
        self.tail_length = 8
        self.auto_delete = False
        self.testing = False
        self.frame_click = False

    def load_from_file(self, file_path):
        try:
            with open(file_path, 'r') as file:
                data = yaml.safe_load(file)
                for key, value in data.items():
                    setattr(self, key, value)
        except Exception as e:
            print(f"Failed to load config from {file_path}, saving default config. Error: {e}")
            self.save_to_file(file_path)

    def save_to_file(self, file_path):
        data = {
            'video': self.video,
            'monitor': self.monitor,
            'yolo': self.yolo,
            'model': self.model,
            'threshold': self.threshold,
            'start_frames': self.start_frames,
            'tail_length': self.tail_length,
            'auto_delete': self.auto_delete,
            'testing': self.testing,
            'frame_click': self.frame_click
        }
        with open(file_path, 'w') as file:
            yaml.safe_dump(data, file)
