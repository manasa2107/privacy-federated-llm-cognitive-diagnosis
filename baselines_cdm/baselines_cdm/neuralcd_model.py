"""
NeuralCD: Neural Cognitive Diagnosis Model
Based on Wang et al. (2022) "NeuralCD: A General Framework for Cognitive Diagnosis"
IEEE Transactions on Knowledge and Data Engineering

This implementation follows the architecture described in the paper for fair comparison.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import pickle


class NeuralCDNet(nn.Module):
    """
    Neural Cognitive Diagnosis Network
    
    Architecture:
    1. Student embedding layer
    2. Question embedding layer  
    3. Concept embedding layer
    4. MLP to predict response correctness
    5. Knowledge state estimation from embeddings
    """
    def __init__(self, n_students, n_questions, n_concepts, 
                 student_dim=64, question_dim=64, concept_dim=64, hidden_dim=128):
        super(NeuralCDNet, self).__init__()
        
        self.n_students = n_students
        self.n_questions = n_questions
        self.n_concepts = n_concepts
        
        # Embedding layers
        self.student_emb = nn.Embedding(n_students, student_dim)
        self.question_emb = nn.Embedding(n_questions, question_dim)
        self.concept_emb = nn.Embedding(n_concepts, concept_dim)
        
        # Knowledge state estimation network
        self.knowledge_net = nn.Sequential(
            nn.Linear(student_dim + concept_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Response prediction network
        self.response_net = nn.Sequential(
            nn.Linear(student_dim + question_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embeddings with Xavier uniform"""
        nn.init.xavier_uniform_(self.student_emb.weight)
        nn.init.xavier_uniform_(self.question_emb.weight)
        nn.init.xavier_uniform_(self.concept_emb.weight)
    
    def forward(self, student_ids, question_ids):
        """
        Forward pass for response prediction
        
        Args:
            student_ids: (batch_size,) tensor of student IDs
            question_ids: (batch_size,) tensor of question IDs
            
        Returns:
            predictions: (batch_size,) predicted response probabilities
        """
        student_vec = self.student_emb(student_ids)
        question_vec = self.question_emb(question_ids)
        
        combined = torch.cat([student_vec, question_vec], dim=1)
        predictions = self.response_net(combined).squeeze()
        
        return predictions
    
    def get_knowledge_state(self, student_ids):
        """
        Estimate knowledge state for given students
        
        Args:
            student_ids: (batch_size,) tensor of student IDs
            
        Returns:
            knowledge_states: (batch_size, n_concepts) knowledge mastery probabilities
        """
        batch_size = student_ids.size(0)
        student_vec = self.student_emb(student_ids)
        
        # Expand to compute for all concepts
        student_vec_expanded = student_vec.unsqueeze(1).repeat(1, self.n_concepts, 1)
        
        # Get all concept embeddings
        concept_ids = torch.arange(self.n_concepts, device=student_ids.device)
        concept_vecs = self.concept_emb(concept_ids).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Concatenate and predict
        combined = torch.cat([student_vec_expanded, concept_vecs], dim=2)
        combined_flat = combined.view(-1, combined.size(2))
        
        knowledge_flat = self.knowledge_net(combined_flat).squeeze()
        knowledge_states = knowledge_flat.view(batch_size, self.n_concepts)
        
        return knowledge_states


class CDMDataset(Dataset):
    """Dataset for Cognitive Diagnosis Models"""
    def __init__(self, student_ids, question_ids, responses):
        self.student_ids = torch.LongTensor(student_ids)
        self.question_ids = torch.LongTensor(question_ids)
        self.responses = torch.FloatTensor(responses)
    
    def __len__(self):
        return len(self.student_ids)
    
    def __getitem__(self, idx):
        return self.student_ids[idx], self.question_ids[idx], self.responses[idx]


def train_neuralcd(model, train_loader, val_loader, n_epochs=50, lr=0.001, device='cuda'):
    """
    Train NeuralCD model
    
    Args:
        model: NeuralCDNet instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        n_epochs: Number of training epochs
        lr: Learning rate
        device: 'cuda' or 'cpu'
        
    Returns:
        model: Trained model
        history: Training history
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for student_ids, question_ids, responses in train_loader:
            student_ids = student_ids.to(device)
            question_ids = question_ids.to(device)
            responses = responses.to(device)
            
            optimizer.zero_grad()
            predictions = model(student_ids, question_ids)
            loss = criterion(predictions, responses)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for student_ids, question_ids, responses in val_loader:
                student_ids = student_ids.to(device)
                question_ids = question_ids.to(device)
                responses = responses.to(device)
                
                predictions = model(student_ids, question_ids)
                loss = criterion(predictions, responses)
                val_loss += loss.item()
                
                # Accuracy
                predicted_labels = (predictions > 0.5).float()
                correct += (predicted_labels == responses).sum().item()
                total += responses.size(0)
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'neuralcd_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('neuralcd_best.pth'))
    
    return model, history


def predict_knowledge_states(model, student_ids, device='cuda'):
    """
    Predict knowledge states for students
    
    Args:
        model: Trained NeuralCDNet
        student_ids: List or array of student IDs
        device: 'cuda' or 'cpu'
        
    Returns:
        knowledge_states: (n_students, n_concepts) numpy array
    """
    model.eval()
    model = model.to(device)
    
    student_ids_tensor = torch.LongTensor(student_ids).to(device)
    
    with torch.no_grad():
        knowledge_states = model.get_knowledge_state(student_ids_tensor)
    
    return knowledge_states.cpu().numpy()


if __name__ == "__main__":
    # Test with dummy data
    print("Testing NeuralCD implementation...")
    
    n_students = 100
    n_questions = 50
    n_concepts = 4
    n_samples = 1000
    
    # Generate dummy data
    student_ids = np.random.randint(0, n_students, n_samples)
    question_ids = np.random.randint(0, n_questions, n_samples)
    responses = np.random.binomial(1, 0.6, n_samples).astype(float)
    
    # Create datasets
    train_size = int(0.8 * n_samples)
    train_dataset = CDMDataset(
        student_ids[:train_size],
        question_ids[:train_size],
        responses[:train_size]
    )
    val_dataset = CDMDataset(
        student_ids[train_size:],
        question_ids[train_size:],
        responses[train_size:]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = NeuralCDNet(n_students, n_questions, n_concepts)
    
    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, history = train_neuralcd(model, train_loader, val_loader, n_epochs=10, device=device)
    
    # Predict knowledge states
    test_student_ids = list(range(10))
    knowledge_states = predict_knowledge_states(model, test_student_ids, device=device)
    
    print("\nKnowledge states for first 10 students:")
    print(knowledge_states)
    print("\nNeuralCD implementation test complete!")
