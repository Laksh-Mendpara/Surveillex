import torch
import torchvision.models as models
import nltk
from nltk.corpus import wordnet as wn
from transformers import BertTokenizer, BertModel

'''model = models.video.r3d_18(pretrained=True)

model.eval()'''
def get_text_embeddings(definitions):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    embeddings = []
    for definition in definitions.values():
        inputs = tokenizer(definition, return_tensors='pt')
        outputs = model(**inputs)
        text_embedding = outputs.last_hidden_state
        embeddings.append(text_embedding)
    print(f"shape of embeddings:{len(embeddings)}")
    return embeddings





'''class MultiModalTransformer(nn.Module):
    def __init__(self, video_dim, text_dim):
        super(MultiModalTransformer, self).__init__()
        self.video_transformer = BertModel.from_pretrained('bert-base-uncased')
        self.text_transformer = BertModel.from_pretrained('bert-base-uncased')
        self.fusion_layer = nn.Linear(video_dim + text_dim, 256)
    
    def forward(self, video_embedding, text_embedding):
        video_features = self.video_transformer(video_embedding)[0]
        text_features = self.text_transformer(text_embedding)[0]
        combined = torch.cat((video_features, text_features), dim=1)
        fused_embedding = self.fusion_layer(combined)
        return fused_embedding

# Example usage
multi_modal_transformer = MultiModalTransformer(video_dim, text_dim)
fused_embedding = multi_modal_transformer(video_embedding, text_embedding)
print("Fused Embedding Shape:", fused_embedding.shape)
'''

if __name__ == '__main__':
    definitions = {'Abuse': 'cruel or inhumane treatment', 'Arrest': 'the act of apprehending (especially apprehending a criminal)', 'Arson': 'malicious burning to destroy property', 'Assault': 'close fighting during the culmination of a military attack', 'Burglary': 'entering a building unlawfully with intent to commit a felony or to steal valuable property', 'Explosion': 'a violent release of energy caused by a chemical or nuclear reaction', 'Fighting': 'the act of fighting; any contest or struggle', 'Normal Videos': 'Regular video content without criminal activity.', 'RoadAccidents': 'Accidents occurring on the road involving vehicles.', 'Robbery': 'larceny by threat of violence', 'Shooting': 'the act of firing a projectile', 'Shoplifting': 'the act of stealing goods that are on display in a store', 'Stealing': 'the act of taking something from someone unlawfully', 'Vandalism': 'willful wanton and malicious destruction of the property of others'}
    embeddings = get_text_embeddings(definitions)
    print (definitions)
    print(embeddings)
