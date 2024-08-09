import torch
import esm
from Bio import SeqIO
from typing import Tuple, List
import matplotlib.pyplot as plt
import numpy as np

def read_sequence(filename: str) -> Tuple[str, str]:
    """Reads the first (reference) sequence from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

def predict_contacts_esm2(sequence: str, model_name: str = "esm2_t33_650M_UR50D") -> torch.Tensor:
    """
    Predicts protein contacts using the specified ESM-2 model.

    Args:
        sequence (str): The protein sequence.
        model_name (str): The name of the ESM-2 model to use.

    Returns:
        torch.Tensor: The predicted contact map.
    """
    # Load the ESM-2 model
    #model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().cuda()
    batch_converter = alphabet.get_batch_converter()

    # Prepare the input
    data = [("protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(next(model.parameters()).device)

    # Predict contacts
    with torch.no_grad():
        results = model.predict_contacts(batch_tokens)

    return results[0].cpu()

def plot_contact_map(contact_map: torch.Tensor, title: str = "Predicted Contact Map"):
    """
    Plots the contact map.

    Args:
        contact_map (torch.Tensor): The predicted contact map.
        title (str): The title for the plot.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(contact_map, cmap='viridis')
    plt.colorbar(label='Contact Probability')
    plt.title(title)
    plt.xlabel('Residue Index')
    plt.ylabel('Residue Index')
    plt.tight_layout()
    plt.savefig('contact_map.png', dpi=300)
    plt.close()

def main():
    # Example usage
    fasta_file = "testing.fasta"
    description, sequence = read_sequence(fasta_file)

    print(f"Predicting contacts for: {description}")
    contact_map = predict_contacts_esm2(sequence)

    print(f"Contact map shape: {contact_map.shape}")
    print("Contact prediction completed.")

    # Plot the contact map
    plot_contact_map(contact_map, title=f"Contact Map for {description}")
    print("Contact map plot saved as 'contact_map.png'")

if __name__ == "__main__":
    main()