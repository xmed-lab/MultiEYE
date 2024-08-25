import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flair import FLAIRModel


class FLAIRMultiLayer(nn.Module):
    def __init__(self, args, device, concept_feat_path, modality='fundus'):
        super().__init__()
        self.modality = modality
        self.raw_concepts = np.load(concept_feat_path).tolist()
        
        self.flair_model = FLAIRModel(device=device, from_checkpoint=True)
        self.latent_dim = self.flair_model.vision_model.proj_dim
        self.concept_classifier = nn.Linear(len(self.raw_concepts), args.n_classes)

        with torch.no_grad():
            text_input_ids, text_attention_mask = self.flair_model.preprocess_text(self.raw_concepts)
            self.embed_concepts = self.flair_model.text_model(text_input_ids, text_attention_mask)
            
        # self.project_1 = nn.Linear(256, self.latent_dim)
        # self.project_2 = nn.Linear(512, self.latent_dim)
        # self.project_3 = nn.Linear(1024, self.latent_dim)
        self.project_4 = nn.Linear(2048, self.latent_dim)
        
    def forward(self, image):
        embed_images = self.flair_model.vision_model(image)
        concept_sim = embed_images @ self.embed_concepts.t()
        sim = self.concept_classifier(concept_sim)
        return sim

    def forward_concept(self, image):
        embed_images = self.flair_model.vision_model(image)
        concept_sim = embed_images @ self.embed_concepts.t()
        sim = self.concept_classifier(concept_sim)
        return sim, concept_sim
    
    def forward_feat(self, image):
        embed_images = self.flair_model.vision_model(image)
        concept_sim = embed_images @ self.embed_concepts.t()
        sim = self.concept_classifier(concept_sim)
        return sim, embed_images
    
    def forward_vis(self, image):
        embed_images, feature_vis = self.flair_model.vision_model.forward_vis(image)
        concept_sim = embed_images @ self.embed_concepts.t()
        sim = self.concept_classifier(concept_sim)
        return sim, feature_vis
    
    def forward_distill(self, image):
        if self.modality == 'oct':
            with torch.no_grad():
                embed_images, inter_1, inter_2, inter_3, inter_4 = self.flair_model.vision_model.forward_inter(image)
        elif self.modality == 'fundus':
            embed_images, inter_1, inter_2, inter_3, inter_4 = self.flair_model.vision_model.forward_inter(image)
        # inter_1 = self.project_1(inter_1)
        # inter_2 = self.project_2(inter_2)
        # inter_3 = self.project_3(inter_3)
        inter_4 = self.project_4(inter_4)
        
        # concept_sim_1 = inter_1 @ self.embed_concepts.t()
        # concept_sim_2 = inter_2 @ self.embed_concepts.t()
        # concept_sim_3 = inter_3 @ self.embed_concepts.t()
        concept_sim_4 = inter_4 @ self.embed_concepts.t()
        concept_sim = embed_images @ self.embed_concepts.t()
        sim = self.concept_classifier(concept_sim)
        # return sim, torch.stack([concept_sim, concept_sim_1, concept_sim_2, concept_sim_3, concept_sim_4], dim=1)
        return sim, torch.stack([concept_sim, concept_sim_4], dim=1)
    
    def get_concepts_feat(self):
        return self.embed_concepts
    