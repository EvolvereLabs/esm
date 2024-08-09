import argparse
import logging
import subprocess
import os
import pandas as pd
import torch
from esm import Alphabet, pretrained
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_name: str) -> Tuple[torch.nn.Module, Alphabet]:
    """Load the pretrained ESM2 model and its alphabet."""
    logging.info(f"Loading model: {model_name}")
    
    # Check if model weights are downloaded
    model_dir = torch.hub.get_dir()
    model_file = os.path.join(model_dir, 'checkpoints', f"{model_name}.pt")
    if not os.path.exists(model_file):
        logging.error(f"Model weights for {model_name} not found in {model_file}. Please ensure the weights are downloaded.")
        raise FileNotFoundError(f"Model weights for {model_name} not found.")
    
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        logging.info("Transferred model to GPU")
    return model, alphabet

def get_git_version() -> Tuple[str, str]:
    """
    Get the current git commit hash and the repository URL.

    Returns:
        Tuple[str, str]: The current git commit hash and the repository URL.
    """
    try:
        # Get the current git commit hash
        git_version = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
        logging.info(f"Git version: {git_version}")
        
        # Get the repository URL
        git_repo = subprocess.check_output(['git', 'config', '--get', 'remote.origin.url']).strip().decode('utf-8')
        logging.info(f"Git repository: {git_repo}")
        
        return git_version, git_repo
    except subprocess.CalledProcessError as e:
        logging.error(f"Error obtaining git version or repository: {e}")
        return "unknown", "unknown"

def encode_sequences(sequences: List[str], alphabet: Alphabet) -> torch.Tensor:
    """Encode sequences using the model's alphabet."""
    batch_converter = alphabet.get_batch_converter()
    data = [("sequence", seq) for seq in sequences]
    _, _, batch_tokens = batch_converter(data)
    return batch_tokens

def compute_log_likelihoods(model: torch.nn.Module, batch_tokens: torch.Tensor, alphabet: Alphabet) -> List[float]:
    """Compute log likelihoods for the given batch of tokens."""
    with torch.no_grad():
        if torch.cuda.is_available():
            batch_tokens = batch_tokens.cuda()
        logits = model(batch_tokens)["logits"]
        logging.info(f"Logits shape: {logits.shape}")
        log_probs = torch.log_softmax(logits, dim=-1)
        logging.info(f"Log probs shape: {log_probs.shape}")
        logging.info(log_probs)
        log_likelihoods = log_probs.gather(2, batch_tokens.unsqueeze(-1)).squeeze(-1)
        logging.info(f"Log likelihoods shape: {log_likelihoods.shape}")
        logging.info(log_likelihoods)
        log_likelihoods = log_likelihoods.sum(dim=1).cpu().numpy()
        logging.info(f"Log likelihoods shape: {log_likelihoods.shape}")
    return log_likelihoods

def main(args):
    # Load the model and alphabet
    model, alphabet = load_model(args.model_name)

    # Read the input CSV file
    logging.info(f"Reading input CSV file: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    sequences = df[args.sequence_column].tolist()

    # Encode the sequences
    logging.info("Encoding sequences")
    batch_tokens = encode_sequences(sequences, alphabet)

    # Compute log likelihoods
    logging.info("Computing log likelihoods")
    log_likelihoods = compute_log_likelihoods(model, batch_tokens, alphabet)

    # Get git version and repository
    git_version, git_repo = get_git_version()

    # Get the script file path
    script_path = os.path.abspath(__file__)
    logging.info(f"Script path: {script_path}")

    # Save the results to a new CSV file
    df["log_likelihood"] = log_likelihoods
    df["git_version"] = git_version
    df["git_repo"] = git_repo
    df["script_path"] = script_path
    df["model_name"] = args.model_name
    logging.info(f"Writing results to output CSV file: {args.output_csv}")
    df.to_csv(args.output_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute ESM2 log likelihoods for sequences in a CSV file.")
    parser.add_argument("--model-name", type=str, required=True, help="Name of the pretrained ESM2 model to use. Models are listed in esm/pretrained.py. Example use: esm2_t33_650M_UR50D")
    parser.add_argument("--input-csv", type=str, required=True, help="Path to the input CSV file containing sequences.")
    parser.add_argument("--sequence-column", type=str, required=True, help="Name of the column containing sequences.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to the output CSV file to save results.")
    args = parser.parse_args()

    # Check if the output path is a CSV file
    if not args.output_csv.lower().endswith('.csv'):
        raise ValueError("The output path must be a CSV file.")

    # Check if the output directory exists, if not, create it
    output_dir = os.path.dirname(args.output_csv)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    main(args)