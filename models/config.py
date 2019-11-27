
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
        self.model_type = 'dialoguegcn' # 'dialoguegcn', 'dialoguegcn','fan' 'acn'
        self.fan_weights_path = './parameters/Resnet18_FER+_pytorch.pth.tar'
        self.face_matching = True
        self.utt_embed_size = 100
        self.att_window_size = 10
        self.lr=0.0005 
        self.l2=0.00001
        self.eval_on_test=True
        self.num_epochs = 1
        self.use_meld_audio=False
        self.use_our_audio=False

