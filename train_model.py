from argparse import ArgumentParser
from pathlib import Path

import torch
from sacrebleu.metrics import BLEU
from tokenizers import Tokenizer
from tqdm import trange

from data import TranslationDataset, SpecialTokens
from decoding import translate
from model import TranslationModel
from torch.utils.data import DataLoader

from tqdm import tqdm


def train_epoch(
    model: TranslationModel,
    train_dataloader,
    optimizer,
    device,
    loss_fn,
    pad_token_src,
    pad_token_tgt  
):
    # train the model for one epoch
    # you can obviously add new arguments or change the API if it does not suit you
    model.train()
    total_loss = 0

    for i, batch in tqdm(enumerate(train_dataloader)):
        if i > 100:
            break
        X, y = torch.tensor(batch[0]), torch.tensor(batch[1])
        X, y = X.to(device), y.to(device)

        # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
        y_input = y[:,:-1]
        y_expected = y[:,1:]
        
        # Get mask to mask out the next words
        sequence_length = y_input.size(1)
        tgt_mask = model.get_tgt_mask(sequence_length).to(device)

        # Standard training except we pass in y_input and tgt_mask
        pred = model(X, y_input, tgt_mask, X==pad_token_src, y_input==pad_token_tgt)

        # Permute pred to have batch size first again
        pred = pred.permute(1, 2, 0)      
        loss = loss_fn(pred, y_expected)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        total_loss += loss.detach().item()
        
    return total_loss / len(train_dataloader)


@torch.inference_mode()
def evaluate(model: TranslationModel, val_dataloader, device, loss_fn, pad_token_src, pad_token_tgt):
    # compute the loss over the entire validation subset
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            X, y = torch.tensor(batch[0]), torch.tensor(batch[1])
            X, y = X.to(device), y.to(device)

            # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
            y_input = y[:,:-1]
            y_expected = y[:,1:]
            
            # Get mask to mask out the next words
            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)

            # Standard training except we pass in y_input and src_mask
            pred = model(X, y_input, tgt_mask, X==pad_token_src, y_input==pad_token_tgt)

            # Permute pred to have batch size first again
            pred = pred.permute(1, 2, 0)      
            loss = loss_fn(pred, y_expected)
            total_loss += loss.detach().item()
        
    return total_loss / len(val_dataloader)


def train_model(data_dir, tokenizer_path, num_epochs):
    src_tokenizer = Tokenizer.from_file(str(tokenizer_path / "tokenizer_de.json"), )
    tgt_tokenizer = Tokenizer.from_file(str(tokenizer_path / "tokenizer_en.json"))

    print("Loading train dataset:")
    train_dataset = TranslationDataset(
        data_dir / "train.de.txt",
        data_dir / "train.en.txt",
        src_tokenizer,
        tgt_tokenizer,
        max_len=128,  # might be enough at first
    )
    print(f"Train dataset size = {len(train_dataset)}")

    print("Loading val dataset:")
    val_dataset = TranslationDataset(
        data_dir / "val.de.txt",
        data_dir / "val.en.txt",
        src_tokenizer,
        tgt_tokenizer,
        max_len=128,
    )
    print(f"Val dataset size = {len(val_dataset)}")
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=32, 
        collate_fn=train_dataset.collate_translation_data
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=32,
        collate_fn=val_dataset.collate_translation_data
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    model = TranslationModel()
    model.to(device)

    # create loss, optimizer, scheduler objects, dataloaders etc.
    # don't forget about collate_fn
    # if you intend to use AMP, you might need something else

    min_val_loss = float("inf")

    opt = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    src_pad_token = src_tokenizer.encode(SpecialTokens.PADDING.value).ids[0]
    tgt_pad_token = tgt_tokenizer.encode(SpecialTokens.PADDING.value).ids[0]

    for epoch in range(1, num_epochs + 1):

        print(f"Start epoche {epoch}!")

        print(f"Training:")
        train_loss = train_epoch(model, train_dataloader, opt, device, loss_fn, src_pad_token, tgt_pad_token)
        print(f"Evaluating:")
        val_loss = evaluate(model, val_dataloader, device, loss_fn, src_pad_token, tgt_pad_token)
        print(f"Train loss: ${train_loss}, Validation loss: {val_loss}")

        # might be useful to translate some sentences from validation to check your decoding implementation

        # also, save the best checkpoint somewhere around here
        if val_loss < min_val_loss:
            print("New best loss! Saving checkpoint")
            torch.save(model.state_dict(), "checkpoint_best.pth")
            min_val_loss = val_loss

        # and the last one in case you need to recover
        # by the way, is this sufficient?
        torch.save(model.state_dict(), "checkpoint_last.pth")

    # load the best checkpoint
    model.load_state_dict(torch.load("checkpoint_best.pth"))
    return model


def translate_test_set(model: TranslationModel, data_dir, tokenizer_path):
    model.eval()

    greedy_translations = []
    with open(data_dir / "test.de.txt") as input_file, open(
        "answers_greedy.txt", "w+"
    ) as output_file:
        # translate with greedy search
        pass

    beam_translations = []
    with open(data_dir / "test.de.txt") as input_file, open(
        "answers_beam.txt", "w+"
    ) as output_file:
        # translate with beam search
        pass

    with open(data_dir / "test.en.txt") as input_file:
        references = [line.strip() for line in input_file]

    bleu = BLEU()
    bleu_greedy = bleu.corpus_score(greedy_translations, [references]).score

    # we're recreating the object, as it might cache some stats
    bleu = BLEU()
    bleu_beam = bleu.corpus_score(beam_translations, [references]).score

    print(f"BLEU with greedy search: {bleu_greedy}, with beam search: {bleu_beam}")
    # maybe log to wandb/comet/neptune as well


if __name__ == "__main__":
    parser = ArgumentParser()
    data_group = parser.add_argument_group("Data paths")
    data_group.add_argument(
        "--data-dir", type=Path, help="Path to the directory containing processed data"
    )
    data_group.add_argument(
        "--tokenizer-path", type=Path, help="Path to the trained tokenizer files"
    )

    # argument groups are useful for separating semantically different parameters
    hparams_group = parser.add_argument_group("Training hyperparameters")
    hparams_group.add_argument(
        "--num-epochs", type=int, default=50, help="Number of training epochs"
    )

    args = parser.parse_args()

    print(Path(args.data_dir).absolute())

    model = train_model(args.data_dir, args.tokenizer_path, args.num_epochs)
    translate_test_set(model, args.data_dir, args.tokenizer_path)
