import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import logging
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
import torch.nn.functional as F

def setup_logging(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logging.basicConfig(
        filename=os.path.join(output_dir, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

class ImprovedMLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.ReLU, dropout=0.2):
        super(ImprovedMLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
                layers.append(nn.BatchNorm1d(sizes[i + 1]))
                layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
        self._init_weights()  # Proper initialization

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _init_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

                    

class SceneAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(SceneAttention, self).__init__()
        self.global_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.size(0)
        query = self.global_query.expand(B, -1, -1)
        scene_out, _ = self.cross_attention(query, x, x)
        return self.norm(scene_out)

# --- Object Attention ---
class ObjectAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(ObjectAttention, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.cross_scene_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, object_features, scene_context):
        obj_attn, _ = self.self_attention(object_features, object_features, object_features)
        scene_attn, _ = self.cross_scene_attention(object_features, scene_context, scene_context)
        combined = obj_attn + scene_attn
        return self.norm(combined)

# --- Graph Interaction ---
class GraphInteraction(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(GraphInteraction, self).__init__()
        self.edge_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, graph_mask=None):
        B, N, D = x.shape
        if graph_mask is not None:
            num_heads = self.edge_attention.num_heads
            graph_mask = graph_mask.unsqueeze(1).repeat(1, num_heads, 1, 1).reshape(B * num_heads, N, N)
            graph_mask = graph_mask.masked_fill(graph_mask == 0, float('-inf')).masked_fill(graph_mask == 1, 0.0)
        else:
            graph_mask = None

        edge_out, _ = self.edge_attention(x, x, x, attn_mask=graph_mask)
        return self.norm(edge_out)

# --- Gated Fusion ---
class GatedFusion(nn.Module):
    def __init__(self, embed_dim):
        super(GatedFusion, self).__init__()
        self.gate = nn.Linear(embed_dim * 3, 3)
        self.proj = nn.Linear(embed_dim * 3, embed_dim)

    def forward(self, scene, obj, graph):
        combined = torch.cat([scene, obj, graph], dim=-1)
        gate_weights = torch.softmax(self.gate(combined), dim=-1)
        fused = gate_weights[..., 0:1] * scene + gate_weights[..., 1:2] * obj + gate_weights[..., 2:3] * graph
        return self.proj(torch.cat([fused, obj, graph], dim=-1))

# --- Similarity Mask Generator ---
def generate_similarity_mask(object_features, threshold=0.8):
    B, N, D = object_features.size()
    normed = F.normalize(object_features, dim=-1)
    sim_matrix = torch.matmul(normed, normed.transpose(1, 2))
    attention_mask = (sim_matrix >= threshold).float()
    return attention_mask

# --- Final Model ---
class EnhancedClipCaptionModel(nn.Module):
    def __init__(self, embed_dim=512, prefix_length=10):
        super(EnhancedClipCaptionModel, self).__init__()
        self.scene_attention = SceneAttention(embed_dim)
        self.object_attention = ObjectAttention(embed_dim)
        self.graph_interaction = GraphInteraction(embed_dim)
        self.gated_fusion = GatedFusion(embed_dim)
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.clip_project = ImprovedMLP((embed_dim, (self.gpt_embedding_size * prefix_length) // 2,
                                     self.gpt_embedding_size * prefix_length))
        self.prefix_length = prefix_length

    def forward(self, tokens, visual_prefix, mask):
        if visual_prefix.dim() == 2:
            visual_prefix = visual_prefix.unsqueeze(1)

        scene_features = self.scene_attention(visual_prefix)
        object_features = self.object_attention(visual_prefix, scene_features)
        graph_mask = generate_similarity_mask(visual_prefix)
        graph_features = self.graph_interaction(visual_prefix, graph_mask=graph_mask)

        scene_expanded = scene_features.expand(-1, object_features.size(1), -1)
        fused_features = self.gated_fusion(scene_expanded, object_features, graph_features)

        fused = fused_features.mean(dim=1)
        prefix_projections = self.clip_project(fused).view(
            -1, self.prefix_length, self.gpt_embedding_size)

        gpt_inputs = self.gpt.transformer.wte(tokens)
        inputs_embeds = torch.cat((prefix_projections, gpt_inputs), dim=1)

        text_mask = mask[:, :tokens.shape[1]]
        new_mask = torch.cat(
            (torch.ones(tokens.size(0), self.prefix_length).to(tokens.device), text_mask), dim=1
        )
        assert new_mask.shape[1] == inputs_embeds.shape[1]

        labels = torch.cat((
            torch.full((tokens.size(0), self.prefix_length), -100).to(tokens.device),
            tokens
        ), dim=1)

        outputs = self.gpt(inputs_embeds=inputs_embeds, attention_mask=new_mask, labels=labels)
        return outputs

class ClipCocoDataset(Dataset):
    def __init__(self, data_path, prefix_length, gpt2_type="gpt2", normalize_prefix=False):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        self.captions_tokens = [torch.tensor(self.tokenizer.encode(cap['caption']), dtype=torch.int64) for cap in captions_raw]
        self.caption2embedding = [cap["clip_embedding"] for cap in captions_raw]
        self.max_seq_len = max(len(t) for t in self.captions_tokens)

    def __len__(self):
        return len(self.captions_tokens)

    def __getitem__(self, idx):
        tokens = self.captions_tokens[idx]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
        mask = tokens.ge(0)
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)
        prefix = self.prefixes[self.caption2embedding[idx]]
        if self.normalize_prefix:
            prefix = prefix / prefix.norm(2, -1)
        return tokens, mask, prefix



def train(dataset, model, args, val_dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.bs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=len(dataloader) * args.epochs)
    output_prefix=args.prefix
    best_val_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_patience = 5
    
    # Training loop
    for epoch in range(args.epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
    
        model.train()
        progress = tqdm(total=len(dataloader), desc=output_prefix)
        epoch_train_loss = 0
        
        for idx, (tokens, mask, prefix) in enumerate(dataloader):
            model.zero_grad()
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            
            outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, dataset.prefix_length - 1: -1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            epoch_train_loss += loss.item()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
            
            # Save checkpoint periodically
            if (idx + 1) % 10000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_latest.pt"),
                )
        
        progress.close()
        avg_train_loss = epoch_train_loss / len(dataloader)

        torch.save(
            model.state_dict(),
            os.path.join(output_dir, f"{output_prefix}_epoch_{epoch + 1}.pt"),
        )
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for tokens, mask, prefix in val_dataloader:
                tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
                outputs = model(tokens, prefix, mask)
                logits = outputs.logits[:, dataset.prefix_length - 1: -1]
                loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        
        # Logging
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        logging.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}_best.pt"),
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        # Early stopping
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='../data/dataBLIP/ViT-Large_train.pkl')
    parser.add_argument('--data_val', default='../dataBLIP/ViT-Large_val.pkl')
    parser.add_argument('--out_dir', default='experiments_result')
    parser.add_argument('--prefix', default='gpt2_BLIP', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--validate_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    args = parser.parse_args()

    setup_logging(args.out_dir)

    logging.info("Starting training with the following arguments:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")

    train_dataset = ClipCocoDataset(args.data, args.prefix_length)
    val_dataset = ClipCocoDataset(args.data_val, args.prefix_length)
    model = EnhancedClipCaptionModel(embed_dim=1024, prefix_length=args.prefix_length)

    train(train_dataset, model, args, val_dataset)


if __name__ == '__main__':
    main()

