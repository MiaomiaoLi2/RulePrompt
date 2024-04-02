# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains the logic for loading data for all TextClassification tasks.
"""

import os
from collections import defaultdict
from openprompt.data_utils.utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor
from utils.data_utils import load_data

class DatasetsProcessor(DataProcessor):

    def __init__(self):
        super().__init__()
        self.labels = []

    def get_examples(self, data_dir, split):     
        text_path = os.path.join(data_dir, f'{split}.txt')
        label_path = os.path.join(data_dir, f'{split}_labels.txt')
        texts = load_data(text_path)
        labels = load_data(label_path)
        labels = [int(l) for l in labels]

        l_count = defaultdict(int)
        for l in labels:
            l_count[l] += 1
        l2d = defaultdict(list)
        for i,label in enumerate(labels):
            l2d[label].append(i)

        examples = []
        for i, text in enumerate(texts):
            example = InputExample(guid=str(i), text_a=text, label=int(labels[i]))
            examples.append(example)
        return examples
    
    def sorted_examples(self, labels, texts):     
        labels = [int(l) for l in labels]

        l_count = defaultdict(int)
        for l in labels:
            l_count[l] += 1
        l2d = defaultdict(list)
        for i,label in enumerate(labels):
            l2d[label].append(i)

        examples = []
        for i, text in enumerate(texts):
            example = InputExample(guid=str(i), text_a=text, label=int(labels[i]))
            examples.append(example)
        return examples
    
    def get_labels(self, data_dir):
        labelname_path = os.path.join(data_dir, f'label_names.txt')
        self.labels = load_data(labelname_path)
        return self.labels
    
    def get_train_texts(self, data_dir):
        text_path = os.path.join(data_dir, f'train.txt')
        return load_data(text_path)
    
PROCESSORS = {
    "Datasets": DatasetsProcessor
}
