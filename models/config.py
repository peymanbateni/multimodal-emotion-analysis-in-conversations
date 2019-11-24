
class Config:
    def __init__(self):
        self.text_in_dim = 768
        self.text_out_dim = 100
        self.context_in_dim = 100
        self.context_out_dim = 100
        self.audio_in_dim = 100
        self.audio_out_dim = 100
        self.vis_in_dim = 100
        self.vis_out_dim = 100
        self.model_type = 'dialoguegcn' # 'dialoguegcn', 'dialoguegcn','fan'
        self.fan_weights_path = './parameters/Resnet18_FER+_pytorch.pth.tar'
        self.utt_embed_size = 100
        self.att_window_size = 10
