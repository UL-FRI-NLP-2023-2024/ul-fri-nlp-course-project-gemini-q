from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset

class QADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        preprompt = "Prosim odgovori na sledeče vprašanje.\n"
        formatted_input = (
            f"<s>[INST]{preprompt}{item['question']}[/INST]{item['answer']}</s>"
        )

        return {
            "input_text": formatted_input,
        }

class QADataloader:
    def __init__(
        self,
        dataset_path,
        batch_size=8,
    ):
        self.dataset_path = dataset_path
        self.batch_size = batch_size

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

        train_dataset = QADataset(train_data)
        test_dataset = QADataset(test_data)

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        return train_dataloader, test_dataloader

if __name__ == "__main__":
    dataset_path = "combine.json"
    qa_dataloader = QADataloader(
        dataset_path,
    )
    train_dataloader, test_dataloader = qa_dataloader.get_dataloaders()

    for batch in train_dataloader:
        print(batch)
        break  # Just to check the first batch
    for batch in test_dataloader:
        print(batch)
        break  # Just to check the first batch

