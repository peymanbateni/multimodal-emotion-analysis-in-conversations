
class Config:
    def __init__(self):
        self.text_in_dim = 300
        self.text_out_dim = 300
        self.context_in_dim = 300
        self.context_out_dim = 300
        self.audio_in_dim = 300
        self.audio_out_dim = 300
        self.vis_in_dim = 300
        self.vis_out_dim = 300
        self.context_window_size = 10
        self.use_dummy = False
        self.utt_embed_size = 400
        self.att_window_size = 5
