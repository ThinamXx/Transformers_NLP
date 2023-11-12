import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from dataset import BilingualDataset, causal_mask
from model import build_transformer

from config import get_config, get_weights_file_path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import warnings

def get_all_sentences(ds, lang):
    """
    Get all sentences from the dataset for the given language.
    """
    for item in ds:
        yield item["translation"][lang]

def get_or_build_tokenizer(config, ds, lang):
    """
    Get or build a tokenizer for the given language.
    """
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2,
        )
        tokenizer.train_from_iterator(
            get_all_sentences(ds, lang),
            trainer,
        )
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_ds(config):
    """
    Get the dataset.
    """
    ds_raw = load_dataset("opus100", f'{config["src_lang"]}-{config["tgt_lang"]}', split="train")
    
    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["src_lang"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["tgt_lang"])
    
    # Keep 90% for training, 10% for validation
    split = ds_raw.train_test_split(0.1)
    ds_train_raw = split["train"]
    ds_val_raw = split["test"]
    
    ds_train = BilingualDataset(ds_train_raw, tokenizer_src, tokenizer_tgt, config["src_lang"], config["tgt_lang"], config["seq_len"])
    ds_val = BilingualDataset(ds_val_raw, tokenizer_src, tokenizer_tgt, config["src_lang"], config["tgt_lang"], config["seq_len"])
    
    max_len_src = 0
    max_len_tgt = 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][config["src_lang"]]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][config["tgt_lang"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    print(f"Max length of source language: {max_len_src}")
    print(f"Max length of target language: {max_len_tgt}")
    
    train_data_loader = DataLoader(ds_train, batch_size=config["batch_size"], shuffle=True,)
    val_data_loader = DataLoader(ds_val, batch_size=1, shuffle=False,)
    
    return train_data_loader, val_data_loader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config["seq_len"], config["d_model"])
    return model

def train_model(config):
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    Path(config["model_dir"]).mkdir(parents=True, exist_ok=True)
    
    train_data_loader, val_data_loader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    # Tensorboard
    writer = SummaryWriter(config["experiment_name"])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], eps=1e-9)
    
    initial_epoch = 0
    global_step = 0
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Loading weights from {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1).to(device)
    
    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_data_loader, desc=f"Processing epoch {epoch:02d}")
        
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device) # (batch_size, seq_len)
            decoder_input = batch["decoder_input"].to(device) # (batch_size, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (batch_size, 1, 1, seq_len)
            decoder_mask = batch["decoder_mask"].to(device) # (batch_size, 1, seq_len, seq_len)
            
            # Run the forward pass
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask) # (batch_size, seq_len, d_model)
            projection_output = model.project(decoder_output) # (batch_size, seq_len, tgt_vocab_size) 
            
            label = batch["label"].to(device) # (batch_size, seq_len)
            
            # (batch_size, seq_len, tgt_vocab_size) --> (batch_size * seq_len, tgt_vocab_size)
            loss = loss_fn(projection_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():.4f}"})
            
            # Log to tensorboard
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()
            
            # Backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            global_step += 1
        
        # Save the model
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, model_filename)
            
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)