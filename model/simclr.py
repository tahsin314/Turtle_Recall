from torch import nn
import timm

class Identity(nn.Module):
        def __init__(self): super().__init__()

        def forward(self, x):
            return x

class ImageEmbedding(nn.Module):       
        
    def __init__(self, model_name='tf_efficientnet_b2', embedding_size=1024):
        super().__init__()
        
        self.backbone = timm.create_model(model_name, pretrained=True, in_chans=3)
        internal_embedding_size = self.backbone._fc.in_features
        self.backbone._fc = ImageEmbedding.Identity()
        
        self.embedding = self.backbone
        
        self.projection = nn.Sequential(
            nn.Linear(in_features=internal_embedding_size, out_features=embedding_size),
            nn.ReLU(),
            nn.Linear(in_features=embedding_size, out_features=embedding_size)
        )

    def calculate_embedding(self, image):
        return self.embedding(image)

    def forward(self, X):
        image = X
        embedding = self.calculate_embedding(image)
        projection = self.projection(embedding)
        return embedding, projection
