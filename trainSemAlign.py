import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from semalignmodel import SemAlign

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def train_semalign_model(model, train_loader, optimizer, num_epochs, device):
    model.train()
    criterion = nn.L1Loss()  # Use L1 loss

    for epoch in range(num_epochs):
        total_loss = 0.0


        for semantic, contexts, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            semantic = semantic.to(device)
            contexts = contexts.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()


            outputs = model(semantic, contexts)


            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    # Parameters
    v_size = 640  # Size of video embeddings
    s_size = 768  # Size of semantic embeddings
    num_epochs = 50
    learning_rate = 0.001


    model = SemAlign(v_size, s_size).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Train the model
    train_semalign_model(model, train_loader, optimizer, num_epochs, device)
