import torch
from sacremoses import MosesDetokenizer, MosesPunctNormalizer
from tokenizers import Tokenizer

from model import TranslationModel
from data import SpecialTokens

# it's a surprise tool that will help you later
detok = MosesDetokenizer(lang="en")
mpn = MosesPunctNormalizer()


def _greedy_decode(
    model: TranslationModel,
    src: torch.Tensor,
    max_len: int,
    src_tokenizer: Tokenizer,
    tgt_tokenizer: Tokenizer,
    device: torch.device,
) -> torch.Tensor:
    """
    Given a batch of source sequences, predict its translations with greedy search.
    The decoding procedure terminates once either max_len steps have passed
    or the "end of sequence" token has been reached for all sentences in the batch.
    :param model: the model to use for translation
    :param src: a (batch, time) tensor of source sentence tokens
    :param max_len: the maximum length of predictions
    :param tgt_tokenizer: target language tokenizer
    :param device: device that the model runs on
    :return: a (batch, time) tensor with predictions

    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    model.eval()

    BOS_token_tgt = tgt_tokenizer.encode(SpecialTokens.BEGINNING.value).ids[0]
    EOS_token_tgt = tgt_tokenizer.encode(SpecialTokens.END.value).ids[0]
    pad_token_src = src_tokenizer.encode(SpecialTokens.PADDING.value).ids[0]
    pad_token_tgt = tgt_tokenizer.encode(SpecialTokens.PADDING.value).ids[0]

    y_input = torch.tensor([[BOS_token_tgt]], dtype=torch.long, device=device)

    for _ in range(max_len):
        # Get source mask
        tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)
        
        pred = model(src, y_input, tgt_mask, src==pad_token_src, y_input==pad_token_tgt)
        
        next_item = pred.topk(1)[1].view(-1)[-1].item() # num with highest probability
        next_item = torch.tensor([[next_item]], device=device)

        # Concatenate previous input with predicted best word
        y_input = torch.cat((y_input, next_item), dim=1)

        # Stop if model predicts end of sentence
        if next_item.view(-1).item() == EOS_token_tgt:
            break

    return y_input.view(-1).tolist()


def _beam_search_decode(
    model: TranslationModel,
    src: torch.Tensor,
    max_len: int,
    tgt_tokenizer: Tokenizer,
    device: torch.device,
    beam_size: int,
) -> torch.Tensor:
    """
    Given a batch of source sequences, predict its translations with beam search.
    The decoding procedure terminates once max_len steps have passed.
    :param model: the model to use for translation
    :param src: a (batch, time) tensor of source sentence tokens
    :param max_len: the maximum length of predictions
    :param tgt_tokenizer: target language tokenizer
    :param device: device that the model runs on
    :param beam_size: the number of hypotheses
    :return: a (batch, time) tensor with predictions
    """
    pass


@torch.inference_mode()
def translate(
    model: torch.nn.Module,
    src_sentences: list[str],
    src_tokenizer: Tokenizer,
    tgt_tokenizer: Tokenizer,
    translation_mode: str,
    device: torch.device,
) -> list[str]:
    """
    Given a list of sentences, generate their translations.
    :param model: the model to use for translation
    :param src_sentences: untokenized source sentences
    :param src_tokenizer: source language tokenizer
    :param tgt_tokenizer: target language tokenizer
    :param translation_mode: either "greedy", "beam" or anything more advanced
    :param device: device that the model runs on
    """
    if translation_mode == "beam":
        raise NotImplementedError()
    
    src = []
    pad_token_src = src_tokenizer.encode(SpecialTokens.PADDING).ids[0].item()
    max_size = 0
    for sentense in src_sentences:
        src_tokens = src_tokenizer.encode(sentense).ids
        src.append(src_tokens)
        max_size = max(max_size, len(src_tokens))
    for i in range(len(src)):
        src[i] = src[i] + [pad_token_src] * (max_size - len(src[i]))
    src = torch.tensor(src).to(device)
    _greedy_decode(model, src, 50, src_tokenizer, tgt_tokenizer, device)
