# RulePrompt

The source code used for paper [RulePrompt: Weakly Supervised Text Classification with Prompting PLMs and Self-Iterative Logical Rules](https://arxiv.org/abs/2403.02932), published in WWW 2024.

## Requirements

Before running, you need to first install the required packages by typing following commands:
```
pip install -r requirements.txt
```
Also, you need to download the stopwords in the NLTK library:
```
import nltk
nltk.download('stopwords')
```

## Dataset

We use four benchmark datasets: AGNews, 20News, NYT-Topics can be found at [here](https://github.com/ZihanWangKi/XClass); and IMDB can be found at [here](https://github.com/yumeng5/LOTClass).
The label names and templates are provided in ```label_names.txt``` and ```manual_template.txt```. 
Only ```train.txt``` will be used as unlabeled text corpus; ```train_labels.txt``` are provided for completeness and evaluation purpose.


## Run the code

We provide scripts to reproduce our results on the benchmark datasets in ```./scripts/```. You can run them in the root directory by:
```
bash ./scripts/20News.sh
```

## Citations

If you find our work useful for your research, please cite the following paper:
```
@inproceedings{li2024ruleprompt,
    title={RulePrompt: Weakly Supervised Text Classification with Prompting PLMs and Self-Iterative Logical Rules}, 
    author={Miaomiao Li and Jiaqi Zhu and Yang Wang and Yi Yang and Yilin Li and Hongan Wang},
    year={2024},
    booktitle={Proceedings of the ACM Web Conference 2024}
}
```
