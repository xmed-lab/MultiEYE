import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flair import FLAIRModel


class FLAIRConceptClassifier(nn.Module):
    def __init__(self, args, device, concept_feat_path):
        super().__init__()
        self.raw_concepts = np.load(concept_feat_path).tolist()
        
        self.flair_model = FLAIRModel(device=device, from_checkpoint=True)
        self.concept_classifier = nn.Linear(len(self.raw_concepts), args.n_classes)
        # torch.nn.init.kaiming_normal_(self.concept_classifier)
        
        text_input_ids, text_attention_mask = self.flair_model.preprocess_text(self.raw_concepts)
        with torch.no_grad():
            self.embed_concepts = self.flair_model.text_model(text_input_ids, text_attention_mask)
        self.latent_dim = self.flair_model.vision_model.proj_dim
        
    def forward(self, image):
        embed_images = self.flair_model.vision_model(image)
        concept_sim = embed_images @ self.embed_concepts.t()
        sim = self.concept_classifier(concept_sim)
        return sim
    
    def forward_feat(self, image):
        embed_images = self.flair_model.vision_model(image)
        concept_sim = embed_images @ self.embed_concepts.t()
        sim = self.concept_classifier(concept_sim)
        return sim, embed_images

    def forward_concept(self, image):
        embed_images = self.flair_model.vision_model(image)
        concept_sim = embed_images @ self.embed_concepts.t()
        sim = self.concept_classifier(concept_sim)
        return sim, concept_sim
    
    def forward_distill(self, image, hidden=False):
        embed_images = self.flair_model.vision_model(image)
        concept_sim = embed_images @ self.embed_concepts.t()
        concept_sim2 = self.embed_concepts @ embed_images.t()
        sim = self.concept_classifier(concept_sim)
        
        if hidden:
            return sim, concept_sim, embed_images
        else:
            return sim, concept_sim, concept_sim2
    
    def get_concepts_feat(self):
        return self.embed_concepts