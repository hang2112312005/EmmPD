import os
import h5py
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel


class BaseDataset(Dataset):
    def __init__(self, args, split, transform=None):
        self.image_dir = args.image_dir
        self.label_file = args.base_dir + 'label_cameylon.csv'
        self.ann_path = args.ann_path
        self.split_path = args.base_dir + 'splits_cameylon_1.csv'
        self.dataset_name = args.dataset_name
        self.n_classes = args.n_classes
        self.max_seq_length = args.max_seq_length
        self.max_fea_length = args.max_fea_length
        self.split = split
        self.transform = transform

        cases = self.clean_data(pd.read_csv(self.split_path).loc[:, self.split].dropna())

        self.examples = []
        root = self.ann_path

        # load labels
        self.labels = self._load_labels()

        BERT_PATH = './ckpts/Bert'
        self.tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
        self.bert = BertModel.from_pretrained(BERT_PATH)
        self.bert.eval()
        existing_ids = set()
        for dir in os.listdir(root):
            if self.dataset_name == 'TCGA':
                dir_id = '.'.join(dir.split('.')[:-1])
            else:
                # dir_id = '-'.join(dir.split('-')[:2])
                dir_id = '.'.join(dir.split('.')[:-1])

            if dir_id in existing_ids:  #
                continue
            existing_ids.add(dir_id)

            if not dir_id in cases.keys():  # check whther contained in the split
                continue
            else:
                img_names = cases[dir_id]

            img_path_list = []
            for img_name in img_names:
                image_path = os.path.join(self.image_dir, img_name)

                if not os.path.exists(image_path + '.h5'):
                    continue
                img_path_list.append(image_path + '.h5')

            if len(img_path_list) == 0:
                print(f"Warning: No .h5 files found for case_id {dir_id}, skipping.")
                continue

            self.examples.append({'id': dir_id, 'image_path': img_path_list, 'split': self.split})

        print(f'The size of {self.split} dataset: {len(self.examples)}')

    def __len__(self):
        return len(self.examples)

    def clean_data(self, data):
        # clean data
        cases = {}
        for idx in range(len(data)):
            case_name = data[idx]
            if self.dataset_name == 'redhouse':
                case_id = '-'.join(case_name.split('-')[:2])
            else:
                case_id = '.'.join(case_name.split('.')[:2])
            if case_id in cases:
                cases[case_id].append(case_name)
            else:
                # new
                cases[case_id] = [case_name]
        return cases


class ImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_paths = example['image_path']
        images = []
        for path in image_paths:
            with h5py.File(path, "r") as f:
                features = torch.tensor(f["features"][:], dtype=torch.float32)
                indices = torch.tensor(f["coords"][:], dtype=torch.int32)
                images.append(features)
        images = torch.cat(images, dim=0)
        label = self.labels.get(image_id, -1)

        if self.dataset_name == 'TCGA':
            text = [
                "adenocarcinomas typically show disorganized glandular structures, marked cellular atypia (enlarged, hyperchromatic, irregular nuclei), abundant mucin production, and frequent mitotic figures. Tumor invasion into deeper tissues is common, accompanied by stromal fibrosis, intraglandular “dirty” necrosis, and tumor budding at the invasive front.",
                "most tumors are conventional adenocarcinomas, characterized by disorganized glandular structures, marked nuclear atypia, mucin production, and stromal desmoplasia. Some tumors are mucinous adenocarcinomas, showing abundant extracellular mucin pools with floating tumor cells, while others are signet-ring cell carcinomas, featuring individual cells with prominent intracytoplasmic mucin pushing the nucleus to the periphery."]

        else:
            text = [
                "negative: Normal lymph node, high resolution: Uniform small lymphocytes densely packed.Well-formed follicles in cortex, lymphocyte cords in medulla.Thin-walled blood vessels throughout.No atypical cells or architectural distor tions.",
                "Micro metastasis：Tumor foci measuring > 0.2 mm and ≤ 2 mm, or clusters with more than 200 cells but less than 2 mm in size.Small, dense clusters of tumor cells forming localized patches.Mild structural disruption present in the surrounding lymphoid tissue.Lymph node architecture partially affected.",
                "Macro metastasis：Tumor foci larger than 2.0 mm.Large tumor cell masses replacing normal lymph node tissue.Marked architectural destruction with irregular borders.Abnormal blood vessels and stromal reactions often present.",
                "Isolated Tumor Cells：Single tumor cells or tiny clusters (≤ 0.2 mm) sparsely scattered in lymphoid tissue.Cells are very small, often appearing as tiny dots or minute aggregates.Minimal contrast difference on H&E staining.Lymph node architecture largely preserved without obvious disruption."
            ]

        tokens = self.tokenizer(text, max_length=512, return_tensors="pt", padding=True, truncation=True)
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feature = outputs.last_hidden_state[:, 0, :]  # shape：[batch_size, seq_len, hidden_dim]

        sample = (image_id, images, label, text_feature)
        return sample

    def _load_labels(self):
        labels = {}
        label_file = self.label_file
        df = pd.read_csv(label_file)
        for idx, row in df.iterrows():
            labels[row['slide_id']] = row['label']
            # image_id = row['slide_id']
            # label = torch.tensor([row[f'disease_{i}'] for i in range(0, self.n_classes)], dtype=torch.float32)
            # labels[image_id] = label
        return labels

    def filter_df(self, df, filter_dict):
        if len(filter_dict) > 0:
            filter_mask = np.full(len(df), True, bool)
            # assert 'label' not in filter_dict.keys()
            for key, val in filter_dict.items():
                mask = df[key].isin(val)
                filter_mask = np.logical_and(filter_mask, mask)
            df = df[filter_mask]
        return df

    def df_prep(self, data, label_dict, ignore, label_col):
        if label_col != 'label':
            data['label'] = data[label_col].copy()

        mask = data['label'].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        for i in data.index:
            key = data.loc[i, 'label']
            data.at[i, 'label'] = label_dict[key]

        return data
