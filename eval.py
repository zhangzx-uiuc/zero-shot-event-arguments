
def safe_div(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0

def compute_f1(predicted, gold, matched):
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def evaluate(pred_res, gold_res, role_type_idxs):
    rev_role_types = {v:k for k,v in role_type_idxs.items()}

    gold_nums = {value:0.0 for key,value in rev_role_types.items()}
    pred_nums = {value:0.0 for key,value in rev_role_types.items()}
    correct_nums = {value:0.0 for key,value in rev_role_types.items()}

    stats = {}

    total_gold_nums, total_pred_nums, total_correct_nums = 0.0, 0.0, 0.0

    unrelated_idx = role_type_idxs["unrelated object"]
    # unrelated_idx = -1

    for i in range(len(pred_res)):
        pred_list_i = pred_res[i]
        gold_list_i = gold_res[i]

        for j in range(len(pred_list_i)):
            # calculate total
            if pred_list_i[j] != unrelated_idx:
                total_pred_nums += 1
                pred_nums[rev_role_types[pred_list_i[j]]] += 1

            if gold_list_i[j] != unrelated_idx:
                total_gold_nums += 1
                gold_nums[rev_role_types[gold_list_i[j]]] += 1

            if pred_list_i[j] == gold_list_i[j] and pred_list_i[j] != unrelated_idx and gold_list_i[j] != unrelated_idx:
                correct_nums[rev_role_types[gold_list_i[j]]] += 1
                total_correct_nums += 1
    
    p, r, f = compute_f1(total_pred_nums, total_gold_nums, total_correct_nums)

    for key in gold_nums:
        pred, gold, matched = pred_nums[key], gold_nums[key], correct_nums[key]
        pi, ri, fi = compute_f1(pred, gold, matched)
        res_dict = {"p":pi, "r":ri, "f":fi}
        stats.update({key: res_dict})
    
    return stats, (p, r, f)
    

            
            
            

