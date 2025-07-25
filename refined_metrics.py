import numpy as np
import argparse
import math
import json



def read_data(file_path):
    alignments = []
    refs = []
    cans = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            alignments.append(obj['alignments_id'])
            refs.append(obj['ref_words'])
            cans.append(obj['can_words'])
    return alignments,refs,cans


def get_key_information(alignments, refs, cans):
    key_alignments = []
    key_refs = []
    key_cans = []
    elements_refs = []
    elements_cans = []
    for alignment, ref, can in zip(alignments, refs, cans):
        elements_ref = list(range(1, len(ref) + 1))  
        elements_can = sorted(list(set([alignment_single[1] for alignment_single in alignment])))

        key_ref = [ref[element-1] for element in elements_ref] 
        key_can = [can[element-1] for element in elements_can] 

        ref_value_map = {old_value: old_value for old_value in elements_ref}
        can_items = sorted(set(sublist[1] for sublist in alignment))
        can_value_map = {old_value: new_value for new_value, old_value in enumerate(can_items, start=1)}

        key_alignment = [[ref_value_map[sublist[0]], can_value_map[sublist[1]]] if sublist[1] in can_value_map else [ref_value_map[sublist[0]], None] for sublist in alignment]
        key_alignment = [item for item in key_alignment if item[1] is not None] 

        elements_refs.append(elements_ref)
        elements_cans.append(elements_can)
        key_refs.append(key_ref)
        key_cans.append(key_can)
        key_alignments.append(key_alignment)
        
    return key_alignments, key_refs, key_cans


def merge_alignments(alignments_list):
    merged_alignments_list = []
    for alignments in alignments_list:
        ref_to_can = {}
        can_to_ref = {}

        for ref, can in alignments:
            if ref not in ref_to_can:
                ref_to_can[ref] = set()
            if can not in can_to_ref:
                can_to_ref[can] = set()
            
            ref_to_can[ref].add(can)
            can_to_ref[can].add(ref)

        visited_ref = set()
        visited_can = set()
        
        def get_all_related(ref):
            related_targets = set()
            stack = [ref]
            while stack:
                current_ref = stack.pop()
                if current_ref in visited_ref:
                    continue
                visited_ref.add(current_ref)
                targets = ref_to_can.get(current_ref, set())
                related_targets.update(targets)
                for t in targets:
                    for s in can_to_ref.get(t, set()):
                        if s not in visited_ref:
                            stack.append(s)
            return related_targets

        def get_all_related_target(can):
            related_sources = set()
            stack = [can]
            while stack:
                current_can = stack.pop()
                if current_can in visited_can:
                    continue
                visited_can.add(current_can)
                sources = can_to_ref.get(current_can, set())
                related_sources.update(sources)
                for s in sources:
                    for t in ref_to_can.get(s, set()):
                        if t not in visited_can:
                            stack.append(t)
            return related_sources

        merged_alignments = []
        for ref in ref_to_can:
            if ref not in visited_ref:
                related_targets = get_all_related(ref)
                for can in related_targets:
                    related_sources = get_all_related_target(can)
                    if related_sources:
                        merged_alignments.append([list(related_sources), list(related_targets)])

        merged_alignments_list.append(merged_alignments)

    return merged_alignments_list


def get_merge_ref_words_len(refs_words, merged_alignments_id):
    refs_words_len = []
    for i, merged_alignment_id in enumerate(merged_alignments_id):
        merged_len = sum((len(sublist[0])-1) if len(sublist[0]) > 1 else 0 for sublist in merged_alignment_id)
        refs_words_len.append(len(refs_words[i]) - merged_len)
    return refs_words_len


def get_ref_can(alignments,refs,cans):
    ref_can = []
    for alignment,ref,can in zip(alignments,refs,cans):
        ref_can_dictnary = []
        for ref_ids, can_ids in alignment:
            ref_combined = ''.join(ref[i-1] for i in ref_ids)
            can_combined = ''.join(can[i-1] for i in can_ids)
            ref_can_dictnary.append([ref_combined, can_combined])
        ref_can.append(ref_can_dictnary)
    return ref_can


import copy

def rearrange_alignments(alignments_list):
    rearranged_alignments = []
    for alignments in alignments_list:
        rearranged_alignment = copy.deepcopy(alignments)
        rearranged_alignment = sorted(rearranged_alignment, key=lambda x: x[0][0])
        ref_count = 0
        ref_short = []
        for alignment in rearranged_alignment:
            ref_short.extend(alignment[0][1:])
            ref_count = sum(1 for x in ref_short if x < alignment[0][0])
            alignment[0] = alignment[0][0]-ref_count
        rearranged_alignment = sorted(rearranged_alignment, key=lambda x: x[1][0])
        for index, item in enumerate(rearranged_alignment):
            item[1] = index + 1
        rearranged_alignment = sorted(rearranged_alignment, key=lambda x: x[0])
        rearranged_alignments.append(rearranged_alignment)
    return rearranged_alignments

def many2one(file_path):
    initial_information = read_data(file_path)
    alignments_id,refs_words,cans_words = initial_information
    key_information = get_key_information(alignments_id,refs_words,cans_words)
    alignments_id,refs_words,cans_words = key_information
    merged_alignments_id = merge_alignments(alignments_id)
    refs_words_len = get_merge_ref_words_len(refs_words, merged_alignments_id)
    ref_can = get_ref_can(merged_alignments_id,refs_words,cans_words)
    rearranged_alignments_id = rearrange_alignments(merged_alignments_id)

    return rearranged_alignments_id, ref_can, refs_words, refs_words_len

def get_refs_id(cans_id,alignments_dic):
    refs_id = 0
    refs_id = alignments_dic.get(cans_id, 0)
    return refs_id

def calculate_bleu(alignments_list,ref_can,refs):
    bleus = []
    for alignments,ref_can_dict,ref in zip(alignments_list,ref_can,refs):
        if ref != ['nan']:
            bleu_sum = 0
            bleu_4 = 0
            len_refs = len(ref) 
            len_cans = len(ref_can_dict) 
            ref_len = sum(len(word) for word in ref)
            can_len = sum(len(ref_can[1]) for ref_can in ref_can_dict)
            bleu = 0  
            BP = 0  
            
            if len_cans >=4:
                n_len = 5
            else:
                n_len = len_cans+1
            
            alignments_dic = {key: value for key, value in alignments}

            for n in range (1,n_len):
                count = 0
                for cans_id in range (1,len_cans - n + 2):
                    en = 0
                    refs_id = -1
                    for i in range (0,n):
                        if get_refs_id(cans_id+i,alignments_dic) == 0 or (refs_id > 0 and get_refs_id(cans_id+i,alignments_dic) != refs_id + 1):
                            en = 1
                            break
                        refs_id = get_refs_id(cans_id+i,alignments_dic)
                    if en == 0:
                        count = count + 1
                if count != 0:
                    bleu_sum = bleu_sum + math.log(count/(len_cans - n + 1))
                else:
                    bleu_sum  = bleu_sum + math.log(1e-9)
            if n_len < 5:
                bleu_sum = bleu_sum + math.log(1e-9 * (5 - n_len))
            bleu_4 = math.exp(bleu_sum/4)

            if ref_len>=can_len:
                BP = math.exp(1-ref_len/can_len)
            elif ref_len<can_len:
                BP = math.exp(1-can_len/ref_len)
            bleu = bleu_4 * BP 
            bleu = round(bleu,4)
            bleus.append(bleu)
        else:
            bleus.append(0)
    return bleus

def impro_bleu_score(file_path):
    
    rearranged_alignments_id, ref_can, refs_words, refs_words_len = many2one(file_path)
    bleus = calculate_bleu(rearranged_alignments_id,ref_can,refs_words)

    return bleus


def calculate_bp(c, r):
    if c <= r:
        return math.exp(1 - r / c)
    else:
        return math.exp(1 - c / r)

def improved_rouge(file_path):

    _, ref_can, ref_words_list, refs_words_len = many2one(file_path)

    total_scores = []
    for ref_can_dict, ref_words, ref_len in zip(ref_can, ref_words_list, refs_words_len):
        if ref_words != ["nan"]:
            score_sum = 0
            for (ref, can) in ref_can_dict:
                c = len(can)
                r = len(ref)
                bp = calculate_bp(c, r)
                match_count = 1
                score_sum += bp * match_count
            total_scores.append(score_sum/ref_len)
        else:
            score_sum = 0
            total_scores.append(score_sum)

    return total_scores

def main():
    parser = argparse.ArgumentParser(description="Process some files.")

    parser.add_argument('--test_model', type=str, default=None)
    
    args = parser.parse_args()

    test_model = args.test_model
    file_path = f'output/alignments_{test_model}.jsonl'
    scores_rouge = improved_rouge(file_path)
    scores_bleu = impro_bleu_score(file_path)
    average_rouge = np.mean(scores_rouge)
    average_bleu = np.mean(scores_bleu)
    print(f'Improved ROUGE score for {test_model}：{average_rouge}')
    print(f'Improved BLEU score for {test_model}：{average_bleu}')

if __name__ == "__main__":
    main()