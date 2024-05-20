from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
from datasets import load_dataset


class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        preprompt = "Prosim odgovori na sledeče vprašanje.\n"
        formatted_input = (
            f"<s>[INST]{preprompt}{item['question']}[/INST]{item['answer']}</s>"
        )

        inputs = self.tokenizer(
            formatted_input,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
        }


class QADataloader:
    def __init__(
        self,
        dataset_path,
        tokenizer_name="mistralai/Mixtral-8x7B-v0.1",
        auth_token=None,
        batch_size=8,
        max_length=512,
    ):
        self.dataset_path = dataset_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=auth_token)

        # Adding special tokens and setting the pad token
        special_tokens_dict = {
            "additional_special_tokens": ["<s>", "[INST]", "[/INST]"]
        }
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        print(f"Added {num_added_toks} special tokens.")

        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.batch_size = batch_size
        self.max_length = max_length

    def load_data(self):
        data = load_dataset("json", data_files=self.dataset_path)["train"]
        return data

    def split_data(self, data, test_size=0.1):
        data_size = len(data)
        test_size = int(data_size * test_size)
        train_size = data_size - test_size
        train_data, test_data = random_split(data, [train_size, test_size])
        return train_data, test_data

    def get_dataloaders(self):
        data = self.load_data()
        train_data, test_data = self.split_data(data)

        train_dataset = QADataset(train_data, self.tokenizer, self.max_length)
        test_dataset = QADataset(test_data, self.tokenizer, self.max_length)

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        return train_dataloader, test_dataloader

    def get_tokenizer(self):
        return self.tokenizer


if __name__ == "__main__":
    dataset_path = "combine.json"
    qa_dataloader = QADataloader(
        dataset_path,
        tokenizer_name="mistralai/Mixtral-8x7B-v0.1",
        auth_token="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",
    )
    train_dataloader, test_dataloader = qa_dataloader.get_dataloaders()

    for batch in train_dataloader:
        print(batch)
        break  # Just to check the first batch
    for batch in test_dataloader:
        print(batch)
        break  # Just to check the first batch
