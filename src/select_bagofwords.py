import os
import re            # re.sub
import pandas as pd  # pd.DataFrame

import bow
import utils

DATA_DIR = "data/"
DATA_FILE = os.path.join(DATA_DIR, "train_text.csv")


def list_to_string(list):
    all_s = ''
    for s in list:
        if all_s == '':
            all_s = s
        else:    
            all_s = all_s + ' ' + s
    return all_s    


def hashtag(tokens_list):
    return re.findall(r'#\w+', tokens_list)


def user(tokens_list):
    return re.findall(r'@\w+', tokens_list)


def count_tokens(tokens):
    """
    ARGUMENT: list of tokens
    RETURN: dictionary {token:count}
    """
    token_count_dict = dict()
    for w in tokens:
        token_count_dict[w] = token_count_dict.get(w,0) + 1
        
    return token_count_dict


def reduced_dict(count_dict, tokens_hashtag, tokens_user, N):
    """
    ARGUMENTS: dictionary of counts {x:count}, N
    RETURN: reduced dictionary of counts, count >= N
    """
    reduced_token_count_dict = dict()
    token_list = []
    
    for w in count_dict:
        if w in tokens_hashtag and count_dict[w] >= 3:
            reduced_token_count_dict[w] = count_dict[w]
            token_list.append(w)
        elif w in tokens_user and count_dict[w] >= 3:
            reduced_token_count_dict[w] = count_dict[w]
            token_list.append(w)            
        elif count_dict[w] >= N:
            reduced_token_count_dict[w] = count_dict[w]
            token_list.append(w)
            
    return reduced_token_count_dict, token_list


# prints dict items in descending count order

def print_token_count(count_dict, N):   
    """
    ARGUMENTS: dictionary of counts {x: count}, N
    prints top N key-value in dictionary
    """
    for w in sorted(count_dict, key = count_dict.get, reverse = True):
        if count_dict[w] >= N:
            #print(f'{w}:{token_count_dict[w]}',sep=' ', end=' ', flush=True)
            print(f'{w}:{token_count_dict[w]}  ', end='\n')
            
            
def main():
    doc = utils.file_to_string(DATA_FILE)
    print(doc[0:1000])

    tokens = bow.clean_doc_text(doc)
    print(tokens[0:20])

    all_tokens = list_to_string(tokens)
    tokens_hashtag = set(hashtag(all_tokens))
    tokens_user = set(user(all_tokens))

    print(all_tokens[0:20])
    print()
    print(len(tokens_hashtag))
    print()
    print(len(tokens_user))

    DATA_TOKEN = os.path.join(DATA_DIR, "tokensfinal/train_text_token6.csv")
    #DATA_TOKEN = os.path.join(DATA_DIR, "tokens_dataset_small/train_text_token.csv")

    token_count_dict = count_tokens(tokens[1:])
    #print(token_count_dict)
    #print('')

    # TOKENS WITH COUNT >= y: x TOKENS
    reduced_token_count_dict, token_list = reduced_dict(token_count_dict, tokens_hashtag, tokens_user, 6)
    utils.list_to_csv(token_list, DATA_TOKEN)
    print(len(token_list))

    print(token_list[0:20])

    print_token_count(reduced_token_count_dict,0)


if __name__ == '__main__':
    main()