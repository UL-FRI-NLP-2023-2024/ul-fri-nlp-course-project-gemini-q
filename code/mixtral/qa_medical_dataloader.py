from torch.utils.data import Dataset, DataLoader
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
        question = item['question']
        context = item['answer']
        inputs = self.tokenizer.encode_plus(
            question,
            context,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'token_type_ids': inputs['token_type_ids'].squeeze(),
        }

class QADataloader:
    def __init__(self, dataset_path, tokenizer_name='bert-base-uncased', batch_size=8, max_length=512):
        self.dataset_path = dataset_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.batch_size = batch_size
        self.max_length = max_length

    def load_data(self):
        data = load_dataset('json', data_files=self.dataset_path)['train']
        return data

    def get_dataloader(self):
        data = self.load_data()
        dataset = QADataset(data, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader

