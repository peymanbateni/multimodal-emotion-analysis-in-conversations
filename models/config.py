class Config:
    def __init__(self, use_texts, use_our_audio, use_meld_audio, num_epochs):
        self.text_in_dim = 768
        self.text_out_dim = 50
        if use_meld_audio:
            self.audio_in_dim = 1611            
        else:
            self.audio_in_dim = 400
        self.audio_out_dim = self.text_out_dim
        self.context_out_dim = 50      
        self.vis_in_dim = 100
        self.vis_out_dim = self.text_out_dim
        self.model_type = 'dialoguegcn' # 'fan', 'dialoguegcn','fan' 'acn'
        self.fan_weights_path = './parameters/Resnet18_FER+_pytorch.pth.tar'
        self.face_matching = True
        self.utt_embed_size = 100
        self.att_window_size = 10
        self.lr=0.0005 
        self.l2=0.00001
        self.eval_on_test=True
        self.num_epochs = num_epochs
        self.use_meld_audio=use_meld_audio
        self.use_our_audio=use_our_audio
        if use_meld_audio == True and use_our_audio == True:
            raise Exception("Can't use both our and MELD audio")
        self.use_clean_audio=False
        self.use_sentiment=False
        self.use_texts=use_texts
        self.use_visual=True
        self.visual_features=True
        #self.save_model=True

