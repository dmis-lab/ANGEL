import os
from tqdm import tqdm
import re
from random import shuffle
import pickle
import copy
import sys
import numpy as np
import pandas as pd
import json
import joblib
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss

def cal_similarity_tfidf(a: list, b: str, vectorizer):
    features_a = vectorizer.transform(a)
    features_b = vectorizer.transform([b])
    features_T = features_a.T
    sim = features_b.dot(features_T).todense()
    return sim[0].argmax()

def byLineReader(filename):
    with open(filename, "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
        # for _ in range(100000):
            yield line
            line = f.readline()
    return

def conv(x):
    if isinstance(x, list) or isinstance(x, set):
        return [conv(xx) for xx in x]
    x = x.strip().lower()
    for ch in ',.;{}[]()+-_*/?!`\"\'=%></':
        x = x.replace(ch, ' ')
    return ' '.join([a for a in x.split() if a])


class UMLS(object):

    def __init__(self, umls_path, source_range=None, lang_range=['ENG'], only_load_dict=False, debug=False):
        self.debug = debug
        self.umls_path = umls_path
        self.source_range = source_range
        self.lang_range = lang_range
        self.detect_type()
        # self.load()
        if not only_load_dict:
            self.load_rel()
            self.load_sty()

    def detect_type(self):
        if os.path.exists(os.path.join(self.umls_path, "MRCONSO.RRF")):
            self.type = "RRF"
        else:
            self.type = "txt"
    
    def generate_name_list_set(self, semantic_type, source_onto):
        name_reader = byLineReader(os.path.join(self.umls_path, "MRCONSO." + self.type))
        semantic_reader = byLineReader(os.path.join(self.umls_path, "MRSTY." + self.type))
        self.cui2pref = dict()
        self.cui_in_onto = set()
        for line in tqdm(name_reader, ascii=True):
            if self.type == "txt":
                l = [t.replace("\"", "") for t in line.split(",")]
            else:
                l = line.strip().split("|")
            cui = l[0]
            lang = l[1]
            source = l[11]
            string = l[14]
            ispref = l[6]
            if lang == "ENG":
                if cui in self.cui2pref:
                    self.cui2pref[cui].append(string)
                else:
                    self.cui2pref[cui] = [string]
                if source in source_onto:
                    self.cui_in_onto.update([cui])
        self.cuis_in_semtc = {}
        for line in tqdm(semantic_reader, ascii=True):
            if self.type == "txt":
                l = [t.replace("\"", "") for t in line.split(",")]
            else:
                l = line.strip().split("|")
            cui = l[0]
            semantic = l[1]
            type_str = l[3].lower()
            if semantic in semantic_type:
                self.cuis_in_semtc[cui] = type_str

        for cui in copy.deepcopy(list(self.cui2pref.keys())):
            if cui not in self.cuis_in_semtc or cui not in self.cui_in_onto:
                self.cui2pref.pop(cui)
        
        syn_count = 0
        for cui in self.cui2pref:
            self.cui2pref[cui] = list(set(conv(self.cui2pref[cui])))
            syn_count += len(self.cui2pref[cui])
        
        print("cui count:", len(self.cui2pref))
        print("synonyms count:", syn_count)
    
    def generate_syn_des(self):
        name_reader = byLineReader(os.path.join(self.umls_path, "MRCONSO." + self.type))
        def_reader = byLineReader(os.path.join(self.umls_path, "MRDEF." + self.type))
        self.cui2description = dict()
        cuiset = set()
        auiset = set()
        for line in tqdm(def_reader, ascii=True):
            if self.type == "txt":
                l = [t.replace("\"", "") for t in line.split(",")]
            else:
                l = line.strip().split("|")
            cui = l[0]
            if cui in self.cui2pref:
                cuiset.update([l[0]])
        for line in tqdm(name_reader, ascii=True):
            if self.type == "txt":
                l = [t.replace("\"", "") for t in line.split(",")]
            else:
                l = line.strip().split("|")
            cui = l[0]
            aui = l[7]
            lang = l[1]
            if lang != 'ENG' and cui in cuiset:
                auiset.update([aui])

        def_reader = byLineReader(os.path.join(self.umls_path, "MRDEF." + self.type))
        for line in tqdm(def_reader, ascii=True):
            if self.type == "txt":
                l = [t.replace("\"", "") for t in line.split(",")]
            else:
                l = line.strip().split("|")
            try:
                cui = l[0]
                aui = l[1]
                defi = l[5].lower()
                if cui in cuiset and aui not in auiset:
                    if cui not in self.cui2description:
                        self.cui2description[cui] = [defi]
                    else:
                        self.cui2description[cui].append(defi)
            except:
                import pdb;pdb.set_trace()
        
        des_count = 0
        for cui in self.cui2description:
            des_count += len(self.cui2description[cui])

        print('number of description:', des_count)
       

tfidf_vectorizer = './umls_tfidf_vectorizer.joblib'
vectorizer = joblib.load(tfidf_vectorizer)

def generate_pair(y, mentions, select_scheme):
    if select_scheme == 'random':
        return random.choice(mentions)
    elif select_scheme == 'sample':
        similarity_estimate = cal_similarity_tfidf(mentions, y, vectorizer)
        # print(similarity_estimate.shape)
        return np.random.choice(mentions, 1, p = similarity_estimate/np.sum(similarity_estimate))[0]
    elif select_scheme == 'most_sim':
        similarity_estimate = cal_similarity_tfidf(mentions, y, vectorizer)
        return mentions[similarity_estimate.argmax()]
    elif select_scheme == 'least_sim':
        similarity_estimate = cal_similarity_tfidf(mentions, y, vectorizer)
        return mentions[similarity_estimate.argmin()]
    else:
        print('Wrong mention selection scheme input!!!')


def generate_neg_pair(a_features, b_features) -> list:
    sim = np.array(b_features.dot(a_features.T).todense())[0]
    top_5_indices = np.argsort(sim)[-5:][::-1]
    return top_5_indices


def find_negative_names(positive_candidates, mention, all_dictionary, all_dict_vectorizer) -> list:
    mention_vector = vectorizer.transform([mention])
    candidate_indices = [all_dictionary.index(candidate) for candidate in positive_candidates]
    mask = np.ones(all_dict_vectorizer.shape[0], dtype=bool)
    mask[candidate_indices] = False
    filtered_features = all_dict_vectorizer[mask]
    filtered_strings = [s for i, s in enumerate(all_dictionary) if mask[i]]
    negative_indices = generate_neg_pair(filtered_features, mention_vector)
    negative_names = [filtered_strings[idx] for idx in negative_indices]

    return negative_names        
        
        
template_sets = ['is defined as', 'is described as', 'are the definations of', 'describe', 'define']
template_sets_nodef = ['are the synonyms of', 'indicate the same concept as', 'has synonyms, such as', 'refers to the same concept as']
template_sets_nosyn = ['is', 'is the same as', 'is', 'is the same as']
def create_line(prefix, mention, context, special_tokens, template_choice):
    if prefix:
        des = ' '.join([special_tokens[0], mention, special_tokens[1], template_choice, context])
    else:
        des = ' '.join([context, template_choice, special_tokens[0], mention, special_tokens[1]])
    return des

def prepare_final_pretraindata(cui2defs, cui2syns, special_tokens = None, select_scheme = 'random'):
    from transformers import BartTokenizer
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    output = []
    for cui in tqdm(cui2syns):
        for syn in cui2syns[cui]:
            if cui not in cui2defs:
                if len(cui2syns[cui]) > 1:
                    pending_set = copy.deepcopy(cui2syns[cui])
                    pending_set.remove(syn)
                    mention = generate_pair(syn, pending_set, select_scheme)
                    random.shuffle(pending_set)
                    idx = random.randint(0, 3)
                    des = create_line(idx>1, mention, ', '.join(pending_set[:3]), special_tokens, template_sets_nodef[idx])
                else:
                    mention = syn
                    idx = random.randint(0, 3)
                    des = create_line(idx>1, mention, syn, special_tokens, template_sets_nosyn[idx])
            else:
                idx = random.randint(0, 4)
                if len(cui2syns[cui]) > 1:
                    pending_set = copy.deepcopy(cui2syns[cui])
                    pending_set.remove(syn)
                    mention = generate_pair(syn, pending_set, select_scheme)
                else:
                    mention = syn
                random.shuffle(cui2defs[cui])
                idx = random.randint(0, 3)
                des = create_line(idx<2, mention, ' '.join(cui2defs[cui][:2]), special_tokens, template_sets[idx])
                tks = tokenizer(des)['input_ids']
                if len(tks) > 700:
                    if idx < 2:
                        des = tokenizer.decode(tks[:700])
                    else:
                        des = tokenizer.decode(tks[-700:])

            output.append([cui, mention, syn, des])
            # print(output[-1])
            # input()
    random.shuffle(output)
    return output
                
### DPO dataset negative sampling 하는 부분                
def prepare_dpo_pretraindata(cui2defs, cui2syns, special_tokens = None, select_scheme = 'random'):
    from transformers import BartTokenizer
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    output = []
    flattened_dict = [value for values in cui2syns.values() for value in values]
    all_feature_vector = vectorizer.fit_transform(flattened_dict)
    # {"prompt": "START dislocation of joint END", "chosen": "dislocation of joint is dislocated joint", "rejected": "dislocation of joint is disorientated"}
    for cui in tqdm(cui2syns):
        for syn in cui2syns[cui]:
            # if random.randint(1, 100) == 1:
                tmp_output = {}
                negative_names = find_negative_names(cui2syns[cui], syn, flattened_dict, all_feature_vector)
                if cui not in cui2defs:
                    if len(cui2syns[cui]) > 1:
                        pending_set = copy.deepcopy(cui2syns[cui])
                        pending_set.remove(syn)
                        mention = generate_pair(syn, pending_set, select_scheme)
                        random.shuffle(pending_set)
                        idx = random.randint(0, 3)
                        des = create_line(idx>1, mention, ', '.join(pending_set[:3]), special_tokens, template_sets_nodef[idx])
                    else:
                        mention = syn
                        idx = random.randint(0, 3)
                        des = create_line(idx>1, mention, syn, special_tokens, template_sets_nosyn[idx])
                else:
                    idx = random.randint(0, 4)
                    if len(cui2syns[cui]) > 1:
                        pending_set = copy.deepcopy(cui2syns[cui])
                        pending_set.remove(syn)
                        mention = generate_pair(syn, pending_set, select_scheme)
                    else:
                        mention = syn
                    random.shuffle(cui2defs[cui])
                    idx = random.randint(0, 3)
                    des = create_line(idx<2, mention, ' '.join(cui2defs[cui][:2]), special_tokens, template_sets[idx])
                    tks = tokenizer(des)['input_ids']
                    if len(tks) > 700:
                        if idx < 2:
                            des = tokenizer.decode(tks[:700])
                        else:
                            des = tokenizer.decode(tks[-700:])

                    tmp_output['prompt'] = des
                    tmp_output['chosen'] = f'{mention} is {syn}'
                    tmp_output['rejected'] = f'{mention} is {random.choice(negative_names)}'
                    
                    output.append(tmp_output)
            # print(output[-1])
            # input()
    random.shuffle(output)
    return output
                
                
if __name__ ==  '__main__':
    semantic_type = set(['T005','T007','T017','T022','T031','T033','T037','T038','T058','T062','T074',
                    'T082','T091','T092','T097','T098','T103','T168','T170','T201','T204'])
    semantic_type_ontology = pd.read_csv('./STY.csv') # TUI->STRING mapping table
    semantic_type_size = 0
    while len(semantic_type)!=semantic_type_size:
        semantic_type_size = len(semantic_type)
        for i in range(len(semantic_type_ontology)):
            if semantic_type_ontology['Parents'][i][-4:] in semantic_type:
                semantic_type.update([semantic_type_ontology['Class ID'][i][-4:]])
    source_onto = ['CPT','FMA','GO','HGNC','HPO','ICD10','ICD10CM','ICD9CM','MDR','MSH','MTH',
                    'NCBI','NCI','NDDF','NDFRT','OMIM','RXNORM','SNOMEDCT_US']
    UMLS = UMLS('./', only_load_dict = True)

    UMLS.generate_name_list_set(semantic_type, source_onto)
    UMLS.generate_syn_des()
    
    print('cuicount', len(UMLS.cui2pref))
    print('defcount', len(UMLS.cui2description))
    count = 0
    for cui in UMLS.cui2pref:
        if len(UMLS.cui2pref[cui]) >=2:
            count += 1
    
    import json
    with open("target_kb.json", "w") as f:
        json.dump(UMLS.cui2pref, f, indent=2)
    
    output = prepare_final_pretraindata(UMLS.cui2description, UMLS.cui2pref, special_tokens = ["START", "END"], select_scheme = 'most_sim')
    # output_4_dpo = prepare_dpo_pretraindata(UMLS.cui2description, UMLS.cui2pref, special_tokens = ["START", "END"], select_scheme = 'most_sim')
    shuffle(output)
    # shuffle(output_4_dpo)
    f1 = None
    # f2 = None
    for i in tqdm(range(len(output))):
        if i%100000 == 0:
            if f1:
                f1.close()
            # if f2:
            #     f2.close()
            f1 = open('./raw_data/data_'+str(i//100000).rjust(3,'0')+'.txt', 'w')
            # f2 = open('./raw_data_dpo/data_'+str(i//100000).rjust(3,'0')+'.txt', 'w')
        f1.write(json.dumps(output[i])+'\n')
        # f2.write(json.dumps(output_4_dpo[i])+'\n')








