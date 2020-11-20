import csv           # csv.reader
import pandas as pd  # pd.DataFrame 


def file_to_string(filepath):
    """
    ARGUMENT: file path
    RETURN: string of text from file
    """
    with open(filepath) as f:
        s = f.read()           # 
    return s                   # string



def list_to_csv(list, filepath):
    """
    ARGUMENT: list, file path
    RETURN: csv file
    Convert to dataframe first, then convert dataframe to csv file
    """
    #df = pd.DataFrame(list, columns=["text_tokens"])
    df = pd.DataFrame(list)
    df.to_csv(filepath, index=False)

#df = pd.DataFrame({'col':L})



def csv_to_list_of_lists(filepath):
    """
    """
    text_list = []
    with open(filepath) as f:
        for row in csv.reader(f):
            text_list.append(row)
    return text_list[1:]    


def csv_to_list_of_strings(filepath):
    """
    """
    text_list = []
    with open(filepath) as f:
        for row in csv.reader(f):
            text_list.append(row[0])
    return text_list[1:]



def dataframe_to_csv(df, filepath):
    csvfile = df.to_csv(filepath, encoding='utf-8', index=False)
    
def csv_to_dataframe(filepath):
    df = pd.read_csv(filepath, header=0)
    return df    

def np_array_to_dataframe(array):
    df = pd.DataFrame(data=array[1:], columns=array[0,0:])
    return df
