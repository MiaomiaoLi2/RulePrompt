import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from transformers import HfArgumentParser, AdamW
from utils.arguments import RulePromptArguments
from utils.contextualize_calibration import calibrate
from utils.simcse_data import load_data_label_desc
from utils.simcse_data import DataUtils, DataUtils2
from utils.model_utils import load_ret_model
from utils.simcse import Model

from openprompt.data_utils.text_classification_dataset import DatasetsProcessor
from openprompt import PromptDataLoader
from openprompt import PromptForClassification
from openprompt.prompts import  EmbVerbalizer, RuleVerbalizer
from openprompt.prompts import ManualTemplate
from openprompt.utils.reproduciblity import set_seed
from openprompt.data_utils.data_processor import InputExample
from openprompt.plms import load_plm

from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from collections import OrderedDict
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score

def get_examples(texts, labels):                       
    labels = [int(l) for l in labels]
    examples = []
    for i, text in enumerate(texts):
        example = InputExample(guid=str(i), text_a=text, label=int(labels[i]))
        examples.append(example)
    return examples

def kmeans_resort(x, cluster_centers_sorted):
    centers = len(cluster_centers_sorted)
    for i in range(centers):
        if x==cluster_centers_sorted[i]:
            return i
    return -1

def del_tensor_ele(x, index):
    x_np = x.cpu().numpy()
    x_np = np.delete(x_np, index)
    x = x_np.tolist()
    return x

def get_confidence_score(similarity_score):
    sorted_tensor, _ = torch.sort(similarity_score, descending=True)      
    max_values = sorted_tensor[:, 0]                                   
    second_max_values = sorted_tensor[:, 1]                              
    confidence_score = max_values - second_max_values 
    return confidence_score

def sort_row(words_row, probs_row): 
    data_to_sort = list(zip(words_row, probs_row)) 
    sorted_data = sorted(data_to_sort, key=lambda x: x[1], reverse=True) 
    sorted_words, sorted_probs = zip(*sorted_data) 
    return sorted_words, sorted_probs 

def get_ssw(all_logits, all_indices, aver_logits, tokenizer):
    print("Getting Strong Signal Words...")
    top100_sorted_logits = all_logits
    top100_sorted_indices = all_indices

    top100_sorted_indices_wostop = [[]] * len(top100_sorted_indices)
    top100_sorted_logits_wostop = [[]] * len(top100_sorted_indices)

    stopwords_vocab = stopwords.words('english')
    for i, word_list in enumerate(top100_sorted_indices):
            delete_idx = []
            for j, word_id in enumerate(word_list):
                word = tokenizer.convert_ids_to_tokens(word_id.tolist())
                word = word[0]
                if word != None:
                    word = word.replace("Ġ", "")
                    word = word.lower()
                if word in stopwords_vocab: 
                    delete_idx.append(j)         
            top100_sorted_indices_wostop[i] = del_tensor_ele(top100_sorted_indices[i], delete_idx)
            top100_sorted_logits_wostop[i] = del_tensor_ele(top100_sorted_logits[i], delete_idx)

    top100_sorted_logits_wostop_new = []
    for i, word_list in enumerate(top100_sorted_logits_wostop):
        result_row = []
        for j, word_id in enumerate(word_list):
            result = top100_sorted_logits_wostop[i][j] / aver_logits[0][top100_sorted_indices_wostop[i][j]]
            result_row.append(result)
        top100_sorted_logits_wostop_new.append(result_row)

    top100_resorted_indices = [] 
    top100_resorted_logits= [] 
    for indices_row, logits_row in zip(top100_sorted_indices_wostop, top100_sorted_logits_wostop_new): 
        sorted_indices2, sorted_logits2 = sort_row(indices_row, logits_row) 
        top100_resorted_indices.append(sorted_indices2) 
        top100_resorted_logits.append(sorted_logits2)

    top100_sorted_words = [[]] * len(top100_resorted_indices)

    for i in range(all_logits.shape[0]):
        top100_sorted_words[i] = tokenizer.convert_ids_to_tokens(top100_resorted_indices[i])
        top100_sorted_words[i] = [s.replace("Ġ", "") for s in  top100_sorted_words[i]]
        top100_sorted_words[i] = [s.lower() for s in top100_sorted_words[i]]

    unique_words_list=[]
    for row in top100_sorted_words:
        unique_dict = OrderedDict()  
        for word in row:
            unique_dict[word] = None
        unique_words = list(unique_dict.keys())[:20]
        unique_words_list.append(unique_words)
    ssw_sorted = [list(words) for words in unique_words_list]
    return ssw_sorted

def sort_lines(train_file, sorted_id):
    selected_lines = [str(train_file[line_number]) for line_number in sorted_id]
    return [s.strip() for s in selected_lines]

def cluster(confidence_score, clu_number):
    confidence_score_numpy = confidence_score.cpu().numpy().reshape(-1,1)
    confidence_score_kmeans = KMeans(n_clusters=clu_number, random_state=0).fit(confidence_score_numpy) #3
    confidence_score_kmeans_labels = confidence_score_kmeans.labels_
    cluster_centers = confidence_score_kmeans.cluster_centers_
    return confidence_score_kmeans_labels, cluster_centers

def frequent_pattern_mining(label, confidence_score, confidence_score_kmeans_labels_sorted, ssw_sorted, support):
    results = {}
    for num in range(2):
        ssw_sorted_label = []
        for i in range(len(confidence_score)):
            if confidence_score_kmeans_labels_sorted[i] == num:
                ssw_sorted_label.append(ssw_sorted[i])
        te = TransactionEncoder()
        te_ary = te.fit(ssw_sorted_label).transform(ssw_sorted_label)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        frequent_itemsets = fpgrowth(df, min_support=support, use_colnames=True, max_len=2)
        filtered_itemset_1 = frequent_itemsets[frequent_itemsets['itemsets'].apply(len) == 1]
        filtered_itemset_2 = frequent_itemsets[frequent_itemsets['itemsets'].apply(len) == 2]
        results["itemsets_"+str(num)+"_"+str(label)+"_1"] = filtered_itemset_1
        results["itemsets_"+str(num)+"_"+str(label)+"_2"] = filtered_itemset_2
    return results

def cluster_and_sort(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(data)
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    cluster_centers_sorted=sorted([i for i in cluster_centers[:,0]],reverse=True)
    cluster_labels_sorted=[kmeans_resort(x,cluster_centers_sorted) for x in cluster_centers[cluster_labels]]
    return cluster_labels_sorted

def Pipline(dataset_length, prompt_model, tokenizer, WrapperClass, mytemplate, dataset,
             rule_prompt_args, all_preds, confidence_score, ssw_sorted, aver_logits, iteration):
    
    data_dict = load_data_label_desc(rule_prompt_args)
    class_num = len(data_dict['label_desc'])

    ## Rule Mining
    print("Rule Mining...")
    all_pred_clusters = {}
    all_results = {}
    for label in np.unique(all_preds):
        print("Obtain rules of categories ", label)
        label_indices = np.where(all_preds == label)[0]
        label_data = confidence_score[label_indices]
        label_clusters = cluster_and_sort(np.array(label_data).reshape(-1, 1), 3)
        all_pred_clusters[label] = label_clusters
        new_ssw_sorted = [ssw_sorted[index] for index in label_indices]
        current_results = frequent_pattern_mining(label, label_data, label_clusters, new_ssw_sorted, rule_prompt_args.support)
        all_results.update(current_results)

    # Disjunctive Sub-Rule
    print("Minging Disjunctive Sub-Rule")
    disjunction_rule_label = []
    disjunction_rule_support = []
    one_itemset_words1_list=[]
    one_itemset_support1_list=[]

    dataframes=[]
    dfs={}
    for i in range(class_num): 
        df_1 = all_results["itemsets_1_"+str(i)+"_1"]
        df_1['itemsets'] = df_1['itemsets'].apply(lambda x: next(iter(x))).astype(str)
        dfs[f'df{i}_1'] = df_1.sort_values(by='support', ascending=False)#.head(10)
        dataframes.append( dfs[f'df{i}_1'])     

    itemsets_list=[]
    for df_itemsets in dataframes:
        itemsets_list.append(df_itemsets['itemsets'].tolist())      
    element_indices = {}
    for idx, item_list in enumerate(itemsets_list):
        for element in item_list:
            if element in element_indices:
                element_indices[element].append(idx)
            else:
                element_indices[element] = [idx]
    elements_to_remove = [element for element, indices in element_indices.items() if len(indices) > 1]

    for i in range(class_num):  
        df_1 = all_results["itemsets_0_"+str(i)+"_1"]
        df_1['itemsets'] = df_1['itemsets'].apply(lambda x: next(iter(x))).astype(str)
        df_x=df_1.sort_values(by='support', ascending=False).head(rule_prompt_args.num_verbalizers)

        df_1=df_1.sort_values(by='support', ascending=False).head(10)
        one_itemset_words1 = df_1['itemsets'].tolist() 

        if len(one_itemset_words1)==2:
            df_y=dfs[f'df{i}_1'].head(rule_prompt_args.num_verbalizers)
            disjunction_rule_label.append(df_y['itemsets'].tolist())
            disjunction_rule_support.append(df_y['support'].tolist())
            one_itemset_words1_list.append(df_y['itemsets'].tolist())
            one_itemset_support1_list.append(df_y['support'].tolist())
        else:
            disjunction_rule_label.append(one_itemset_words1)
            disjunction_rule_support.append(df_1['support'].tolist())
            one_itemset_words1_list.append(df_x['itemsets'].tolist())
            one_itemset_support1_list.append(df_x['support'].tolist())

    # Conjunctive Sub-Rule
    print("Minging Conjunctive Sub-Rule")
    itemset_2_1 = []
    itemset_2_2 = []

    conjunction_rule_path_label = []
    conjunction_rule_path_support = []
    conjunction_rule_word1 = []
    conjunction_rule_word2 = []
    conjunction_rule_word3 = []
    conjunction_rule_word4 = []
    conjunction_rule_support = []
    weight1 = [[]]*class_num  
    weight2 = [[]]*class_num  

    for i in range(class_num):  
        df_con_1 = all_results["itemsets_1_"+str(i)+"_1"]
        df_con_1=df_con_1.sort_values(by='support', ascending=False)

        con_one_itemset_words = df_con_1['itemsets'].tolist()
        con_one_itemset_support = df_con_1['support'].tolist()
        conjunction_rule_path_label.append(con_one_itemset_words)
        conjunction_rule_path_support.append(con_one_itemset_support)

        df_con_2 = all_results["itemsets_1_"+str(i)+"_2"]
        df_con_2 = pd.DataFrame({
            'support': df_con_2['support'],
            'itemsets1': df_con_2['itemsets'].apply(lambda x: tuple(x)[0]),
            'itemsets2': df_con_2['itemsets'].apply(lambda x: tuple(x)[1])
        })

        itemset_2_1.append(df_con_2['itemsets1'].tolist())
        itemset_2_2.append(df_con_2['itemsets2'].tolist())

        refiltered_df_two_itemsets = df_con_2[~df_con_2['itemsets1'].str.contains('|'.join(elements_to_remove))] #, regex=False
        refiltered_df_two_itemsets = refiltered_df_two_itemsets[~refiltered_df_two_itemsets['itemsets2'].str.contains('|'.join(elements_to_remove))]
        refiltered_df_two_itemsets = refiltered_df_two_itemsets.sort_values(by='support', ascending=False).head(10)

        words1_list = refiltered_df_two_itemsets['itemsets1'].tolist()
        words2_list = refiltered_df_two_itemsets['itemsets2'].tolist()
        conjunction_rule_word1.append(words1_list)
        conjunction_rule_word2.append(words2_list)
        conjunction_rule_support.append(refiltered_df_two_itemsets['support'].tolist())

        weight_list=[]
        for word in words1_list:
            position = con_one_itemset_words.index(word)
            weight_list.append(con_one_itemset_support[position])
        weight1[i]= weight_list
        weight_list=[]
        for word in words2_list:
            position = con_one_itemset_words.index(word)
            weight_list.append(con_one_itemset_support[position])
        weight2[i]=weight_list
        for j in range(len(weight1[i])):
            weight1[i][j] = float(weight1[i][j])
            weight2[i][j] = float(weight2[i][j])
            weight1[i][j] = weight1[i][j] / (weight1[i][j]+weight2[i][j])
            weight2[i][j] = 1-weight1[i][j]

        conjunction_rule_word3_list=[]
        conjunction_rule_word4_list=[]
        for j in range(int(len(words1_list)/2)):
            k=2*j
            if j % 2 ==0:
                conjunction_rule_word3_list.append(words1_list[k])
                conjunction_rule_word3_list.append(words2_list[k])
                conjunction_rule_word4_list.append(words1_list[k+1])
                conjunction_rule_word4_list.append(words2_list[k+1])
            else:
                conjunction_rule_word3_list.append(words1_list[k+1])
                conjunction_rule_word3_list.append(words2_list[k+1])
                conjunction_rule_word4_list.append(words1_list[k])
                conjunction_rule_word4_list.append(words2_list[k])
        conjunction_rule_word3.append(conjunction_rule_word3_list)
        conjunction_rule_word4.append(conjunction_rule_word4_list)

    ## Rule-Enhanced Pseudo Label Generation
    print("Rule-Enhanced Pseudo Label Generation...")
    
    max_dis_similarity_score_list = [[]] *class_num  #label*text_len   
    max_con_similarity_score_list = [[]] *class_num  #label*text_len   

    # Embedding-based Similarity Matching Unit

    ret_model, ret_tokenizer = load_ret_model(rule_prompt_args, dtype=torch.bfloat16)
    wrap_ret_model = Model(ret_model, ret_tokenizer, rule_prompt_args) # sentence encoder model

    for i in range(class_num):     
        data_dict['label_desc'] = disjunction_rule_label[i]
        data_utils = DataUtils(data_dict, rule_prompt_args)
        _, _, _, similarity_score = data_utils.generate_pseudo_label(wrap_ret_model) 
        max_similarity_score2=[]
        for j, row in enumerate(similarity_score):
            max_similarity_score =0
            for k in range(len(row)):
                max_similarity_score += row[k]* disjunction_rule_support[i][k]
            max_similarity_score2.append(max_similarity_score/len(row))
        max_dis_similarity_score_list[i]=max_similarity_score2
    
    for i in range(class_num):    
        if(len(conjunction_rule_word1[i])==0):
            max_con_similarity_score_list[i] = [0]*len(similarity_score)
        else:
            data_dict['label_desc'] = conjunction_rule_word1[i]
            data_dict['label_desc2'] = conjunction_rule_word2[i]
            data_utils = DataUtils2(data_dict, rule_prompt_args, weight1[i], weight2[i])        
            _, _, _, similarity_score = data_utils.generate_pseudo_label2(wrap_ret_model) 
            max_similarity_score2=[]
            for j, row in enumerate(similarity_score):
                max_similarity_score =0
                for k in range(len(row)):
                    max_similarity_score += row[k]* conjunction_rule_support[i][k]
                max_similarity_score2.append(max_similarity_score/len(row))
            max_con_similarity_score_list[i] = max_similarity_score2

    # Word Overlapping- based Similarity Matching Unit

    disjunction_rule_label_add_and = []
    conjunction_rule_label_add_and3 = []
    conjunction_rule_label_add_and4 = []
    for i in range(class_num):  
        disjunction_rule_label_add_and.append(" and ".join(disjunction_rule_label[i]))
        conjunction_rule_label_add_and3.append(" and ".join(conjunction_rule_word3[i]))
        conjunction_rule_label_add_and4.append(" and ".join(conjunction_rule_word4[i]))

    dataset['rule'] = get_examples(disjunction_rule_label_add_and+conjunction_rule_label_add_and3+conjunction_rule_label_add_and4, list(range(len(disjunction_rule_label_add_and)))+list(range(len(disjunction_rule_label_add_and)))+list(range(len(disjunction_rule_label_add_and))))

    rule_dataloader = PromptDataLoader(dataset=dataset['rule'], template=mytemplate, tokenizer=tokenizer,
                                    tokenizer_wrapper_class=WrapperClass, max_seq_length=rule_prompt_args.max_len, decoder_max_length=3,
                                    batch_size=rule_prompt_args.batch_s, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                    truncate_method="tail")
    rule_all_logits, rule_all_indices, _, _ = calibrate(dataset_length,prompt_model, rule_dataloader)

    ssw = get_ssw(rule_all_logits, rule_all_indices, aver_logits, tokenizer)

    intersection_count = [[]] * len(ssw_sorted)
    for i in range(len(ssw_sorted)):
        ssw_sorted_set = set(ssw_sorted[i])
        inter_list=[]
        for j in range(len(ssw)):
            ssw_set = set(ssw[j])
            inter_list.append(len(ssw_set.intersection(ssw_sorted_set))/20)
        intersection_count[i]=inter_list
    
    intersection_count_1 = [row[0:class_num] for row in intersection_count]
    intersection_count_2 = [row[class_num:class_num*2] for row in intersection_count]
    intersection_count_3 = [row[class_num*2:class_num*3] for row in intersection_count]

    min_list = [[max(x, y) for x, y in zip(row1, row2)] for row1, row2 in zip(intersection_count_2, intersection_count_3)]
    intersection_count_result = [[x + y for x, y in zip(row1, row2)] for row1, row2 in zip(intersection_count_1, min_list)]

    intersection_count2=[]
    for row in intersection_count_result:
        exp_row = np.exp(row - np.max(row)) 
        softmax_row = exp_row / exp_row.sum()
        intersection_count2.append(softmax_row)

    intersection_count=intersection_count2

    # Verbalizer-based Category Estimation Unit

    data_dict = load_data_label_desc(rule_prompt_args)

    plm = prompt_model.plm.to("cpu")
    myverbalizer2 = RuleVerbalizer(one_itemset_words1_list, one_itemset_support1_list, tokenizer, model=plm, classes=data_dict['label_desc']).from_file(
            select_num=rule_prompt_args.select, path=f'{rule_prompt_args.dataset}_{rule_prompt_args.model}_cos.pt', tmodel=rule_prompt_args.model)
    prompt_model2 = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer2, freeze_plm=False)
    prompt_model2 = prompt_model2.to(rule_prompt_args.device)
    
    train_dataloader2 = PromptDataLoader(dataset=dataset['mining_rule'], template=mytemplate, tokenizer=tokenizer,
                                    tokenizer_wrapper_class=WrapperClass, max_seq_length=rule_prompt_args.max_len, decoder_max_length=3,
                                    batch_size=rule_prompt_args.batch_s, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                    truncate_method="tail")
    
    
    plm_similarity_score = []
    for i, batch in enumerate(tqdm(train_dataloader2, desc='Verbalizer-based Category Estimation')):
        batch = batch.to(prompt_model2.device)
        outputs, _ = prompt_model2(batch)  
        plm_similarity_score.append(outputs.cpu())
    plm_similarity_score2 = torch.cat(plm_similarity_score, dim=0)
    total_similarity_score2 = torch.softmax(plm_similarity_score2, dim=-1)

    # Average together

    new_pred=[]
    new_plm_similarity_score = [] 
    for j in range(len(similarity_score)):
        max_similarity_score2=[]
        for i in range(class_num):   
            max_similarity_score2.append(max(total_similarity_score2[j][i] + max_dis_similarity_score_list[i][j]+intersection_count2[j][i], total_similarity_score2[j][i] + max_con_similarity_score_list[i][j]+intersection_count2[j][i])) #
        new_plm_similarity_score.append(max_similarity_score2)
        new_pred.append(max_similarity_score2.index(max(max_similarity_score2)))

    new_plm_similarity_score = torch.tensor(new_plm_similarity_score)

    # majority of texts for Self-Supervised Fine-Tuning
    sorted_id = sorted(range(len(new_pred)), key=lambda k: new_pred[k], reverse=True)
    sorted_len = int(rule_prompt_args.proportion_ft*len(similarity_score))
    sorted_id = sorted_id[:sorted_len]
    
    sorted_labels = sort_lines(new_pred, sorted_id = sorted_id)
    sorted_texts = sort_lines(dataset['train_text'], sorted_id = sorted_id)

    # Evaluation of current iteration
    print("Evaluation of current iteration.")
    file_path = rule_prompt_args.data_dir+'/train_labels.txt'  
    true_labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            value = line.strip()
            true_labels.append(int(value))

    accuracy = accuracy_score(true_labels, new_pred)
    macro_f1 = f1_score(true_labels, new_pred, average='macro')
    micro_f1 = f1_score(true_labels, new_pred, average='micro')
    print("Accuracy: ", accuracy)
    print("Macro F1 Score: ", macro_f1)
    print("Micro F1 Score: ", micro_f1)

    result_path = './output/'+str(rule_prompt_args.dataset)+'_results.txt'
    with open(result_path, "a") as file:
        file.write("Iterations Num: " + str(iteration)+"\n")
        file.write("Accuracy: " + str(accuracy)+"\n")
        file.write("Macro F1 Score: " + str(macro_f1)+"\n")
        file.write("Micro F1 Score: " + str(micro_f1)+"\n")
        file.write("\n")
        file.close()

    return prompt_model2, new_pred, new_plm_similarity_score, sorted_labels, sorted_texts


def main():

    set_seed(100)

    parser = HfArgumentParser((RulePromptArguments))
    (rule_prompt_args,) = parser.parse_args_into_dataclasses()
    print('Dataset Name:', rule_prompt_args.dataset)
    plm, tokenizer, _, WrapperClass = load_plm(rule_prompt_args.model, rule_prompt_args.model_name_or_path)
    dataset = {
        'mining_rule': DatasetsProcessor().get_train_examples(rule_prompt_args.data_dir),
        'train_text': DatasetsProcessor().get_train_texts(rule_prompt_args.data_dir)
    }
    dataset_length = len(dataset['mining_rule'])
    class_labels = DatasetsProcessor().get_labels(rule_prompt_args.data_dir)  # only for completeness and evaluation
    myverbalizer = EmbVerbalizer(tokenizer, model=plm, classes=class_labels).from_file(
            select_num=rule_prompt_args.select, path=f'{rule_prompt_args.dataset}_{rule_prompt_args.model}_cos.pt', tmodel=rule_prompt_args.model)
    mytemplate = ManualTemplate(tokenizer=tokenizer).from_file(f'{rule_prompt_args.data_dir}/manual_template.txt')
    
    with torch.no_grad():
        print("Iterations Num: 1")

        # PLM model
        prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
        prompt_model = prompt_model.to(rule_prompt_args.device)

        # rule-mining dataloader
        mining_dataloader = PromptDataLoader(dataset=dataset['mining_rule'], template=mytemplate, verbalizer=myverbalizer, tokenizer=tokenizer,
                                        tokenizer_wrapper_class=WrapperClass, max_seq_length=rule_prompt_args.max_len, decoder_max_length=3,
                                        batch_size=rule_prompt_args.batch_s, shuffle=False, teacher_forcing=False, predict_eos_token=False, truncate_method="tail")
        
        # ssw and label
        all_preds = []
        all_logits, all_indices, aver_logits, label_logits = calibrate(dataset_length, prompt_model, mining_dataloader)
        label_probs = torch.softmax(label_logits, dim=-1)
        all_preds.extend(torch.argmax(label_probs, dim=-1).tolist())

        # confidence score
        confidence_score = get_confidence_score(label_probs)
        ssw_sorted = get_ssw(all_logits, all_indices, aver_logits, tokenizer)
    
        prompt_model_verb, new_pred, new_plm_similarity_score, sorted_labels, sorted_texts = Pipline(
             dataset_length, prompt_model, tokenizer, WrapperClass, mytemplate, dataset, 
             rule_prompt_args, all_preds, confidence_score, ssw_sorted, aver_logits, iteration=1)
        
    # Train
    loss_func = torch.nn.CrossEntropyLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    for iteration in range(2, rule_prompt_args.num_iterations+1):
        print("Iterations Num: " + str(iteration))
        prompt_model_verb.to(rule_prompt_args.device)
        optimizer_grouped_parameters = [
            {'params': [p for n, p in prompt_model_verb.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0001},
            {'params': [p for n, p in prompt_model_verb.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=rule_prompt_args.learning_rate)

        dataset['train'] = DatasetsProcessor().get_sorted_examples(sorted_labels, sorted_texts)
        train_dataloader = PromptDataLoader(dataset=dataset['train'], template=mytemplate, tokenizer=tokenizer,
                                    tokenizer_wrapper_class=WrapperClass, max_seq_length=rule_prompt_args.max_len, decoder_max_length=3,
                                    batch_size=rule_prompt_args.batch_s, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                    truncate_method="tail")
        
        for epoch in range(rule_prompt_args.epochs): 
            for step, inputs in enumerate(tqdm(train_dataloader)):
                inputs = inputs.to(prompt_model_verb.device)
                logits, _ = prompt_model_verb(inputs) 
                log_probs = torch.nn.functional.softmax(logits, dim=1)
                loss = loss_func(logits, log_probs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()  
                if step %100 ==1:
                    print("Epoch {}, average loss: {}".format(epoch, loss), flush=True)
            
        with torch.no_grad():
            all_preds_train = []
            all_logits, all_indices, aver_logits, label_logits = calibrate(dataset_length, prompt_model_verb, mining_dataloader)
            label_probs = torch.softmax(label_logits, dim=-1)
            all_preds_train.extend(torch.argmax(label_probs, dim=-1).tolist())
            
            ssw_sorted = get_ssw(all_logits, all_indices, aver_logits, tokenizer)
            new_confidence_score = get_confidence_score(new_plm_similarity_score)

            prompt_model_verb, new_pred, new_plm_similarity_score, sorted_labels, sorted_texts = Pipline(
                 dataset_length, prompt_model_verb, tokenizer, WrapperClass, mytemplate, dataset, 
                 rule_prompt_args, new_pred, new_confidence_score, ssw_sorted, aver_logits, iteration)


if __name__ == '__main__':
    main()