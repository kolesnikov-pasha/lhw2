from enum import Enum
from pathlib import Path
import bs4 as bs

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.data import Dataset
from tqdm import tqdm


class SpecialTokens(Enum):
    UNKNOWN = "[UNK]"
    PADDING = "[PAD]"
    BEGINNING = "[BOS]"
    END = "[EOS]"


def process_training_file(input_path: Path, output_path: Path):
    result = {}
    with open(input_path, "r") as source:
        html = bs.BeautifulSoup(source)
        for doc in html.find_all("doc"):
            try:
                text = list(filter(lambda s: s != None and s != '\n' and not str(s).startswith("<"), doc.contents))[0]
            except:
                text = ""
            result[int(doc.talkid.string)] = text
    max_index = max(list(result.keys()))
    with open(output_path, "w") as output:
        for i in range(max_index):
            texts = result.get(i + 1, "").split("\n")
            for text in texts:
                output.write(f"{SpecialTokens.BEGINNING.value} {text.strip()} {SpecialTokens.END.value}\n")

def process_evaluation_file(input_path: Path, output_path: Path):
    result = []
    with open(input_path, "r") as source:
        html = bs.BeautifulSoup(source)
        for doc in html.find_all("doc"):
            for seg in doc.find_all("seg"):
                result.append(f"{SpecialTokens.BEGINNING.value} {seg.string} {SpecialTokens.END.value}\n")
    with open(output_path, "w") as output:
        output.writelines(result)


def convert_files(base_path: Path, output_path: Path):
    """
    Given a directory containing all the dataset files, convert each one into the "one example per line" format.
    :param base_path: Path containing files with original data
    :param output_path: Path containing files with processed data
    """

    for language in "de", "en":
        process_training_file(
            base_path / f"train.tags.de-en.{language}",
            output_path / f"train.{language}.txt",
        )
        process_evaluation_file(
            base_path / f"IWSLT17.TED.dev2010.de-en.{language}.xml",
            output_path / f"val.{language}.txt",
        )
        process_evaluation_file(
            base_path / f"IWSLT17.TED.tst2010.de-en.{language}.xml",
            output_path / f"test.{language}.txt",
        )


class TranslationDataset(Dataset):
    def __init__(
        self,
        src_file_path,
        tgt_file_path,
        src_tokenizer: Tokenizer,
        tgt_tokenizer: Tokenizer,
        max_len=32,
    ):
        """
        Loads the training dataset and parses it into separate tokenized training examples.
        No padding should be applied at this stage
        :param src_file_path: Path to the source language training data
        :param tgt_file_path: Path to the target language training data
        :param src_tokenizer: Trained tokenizer for the source language
        :param tgt_tokenizer: Trained tokenizer for the target language
        :param max_len: Maximum length of source and target sentences for each example:
        if either of the parts contains more tokens, it needs to be filtered.
        """
        with open(src_file_path, "r") as src:
            src_lines = src.readlines()
        with open(tgt_file_path, "r") as tgt:
            tgt_lines = tgt.readlines()
        self.examples = []
        for i in tqdm(range(min(len(src_lines), len(tgt_lines)))):
            src = src_lines[i]
            tgt = tgt_lines[i]
            if len(src) == 0 or len(tgt) == 0:
                continue
            src_tokens = src_tokenizer.encode(src).ids
            tgt_tokens = tgt_tokenizer.encode(tgt).ids
            if len(src_tokens) > max_len or len(tgt_tokens) > max_len:
                continue
            self.examples.append((src_tokens, tgt_tokens))

        self.end_src = src_tokenizer.encode(SpecialTokens.END.value).ids[0]
        self.pad_src = src_tokenizer.encode(SpecialTokens.PADDING.value).ids[0]
        self.end_tgt = tgt_tokenizer.encode(SpecialTokens.END.value).ids[0]
        self.pad_tgt = tgt_tokenizer.encode(SpecialTokens.PADDING.value).ids[0]
        self.max_len = max_len


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

    def collate_translation_data(self, batch):
        """
        Given a batch of examples with varying length, collate it into `source` and `target` tensors for the model.
        This method is meant to be used when instantiating the DataLoader class for training and validation datasets in your pipeline.
        """
        source = []
        target = []
        for example in batch:
            src = example[0] + (self.max_len - len(example[0])) * [self.pad_src]
            tgt = example[1] + (self.max_len - len(example[1])) * [self.pad_tgt]
            source.append(src)
            target.append(tgt)
        return source, target


def train_tokenizer(train_file: Path, validation_file: Path, save_dir):
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], show_progress=True, vocab_size=30000)
    tokenizer.train(files=[str(train_file.absolute()), str(validation_file.absolute())], trainer=trainer)
    tokenizer.save(str(save_dir))


def train_tokenizers(base_dir: Path, save_dir: Path):
    """
    Trains tokenizers for source and target languages and saves them to `save_dir`.
    :param base_dir: Directory containing processed training and validation data (.txt files from `convert_files`)
    :param save_dir: Directory for storing trained tokenizer data (two files: `tokenizer_de.json` and `tokenizer_en.json`)
    """
    train_tokenizer(base_dir / "train.de.txt", base_dir / "val.de.txt", save_dir / "tokenizer_de.json")
    train_tokenizer(base_dir / "train.en.txt", base_dir / "val.en.txt", save_dir / "tokenizer_en.json")
