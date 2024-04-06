import datasets
import os
from typing import List, Tuple
import numpy as np
from jinja2 import Template
import pandas as pd

np.random.seed(42)


def import_prompts(folder_path:str) -> List[str]:
    """
    :params: Folder path - path to the folder from current directory.

    This is to import hard-coded prompts that we have made in folders.
    """
    text_list = []
    if not os.path.isdir(folder_path):
        print("Error: The provided path is not a directory.")
        return text_list
    
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.txt'):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    text_list.append(text)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
    
    return text_list

def import_templates(folder_path: str, toxicity_prompt_list: List[str]) -> List:

    template_list = [Template(s) for s in import_prompts(folder_path)]

    jailbreak_prompts = []

    for i, prompt in enumerate(toxicity_prompt_list):

        template = np.random.choice(template_list)
        jailbreak_prompts.append(template.render({'prompt': prompt}))

    return jailbreak_prompts



def main(split='train',jailbreak=False) -> None:
    """
    We generate two datasets, one for us to 'unlearn' from and another for us to use as a test, 
    where we have a mixture of jailbreaking attempts and entries from toxicity chat to unlearn from. 
    """

    print("import dataset")
    tc = datasets.load_dataset('lmsys/toxic-chat', 'toxicchat0124', split=split)
    toxic_tc = tc.filter(lambda x: x['toxicity']==1)
    non_jailbreak_toxic_tc = toxic_tc.filter(lambda x: x['jailbreaking']==0)

    if jailbreak:
        print("importing hardcoded prompts")
        hardcoded_jailbreaks = import_prompts('toxic_prompts')

        print("importing templated prompts")
        jailbreak_list = import_templates('prompt_templates',non_jailbreak_toxic_tc['user_input'])

        # makes us a list of inputs for doing any datastuff with, along with the toxicity labels.
        inputs = tc['user_input'] + jailbreak_list + hardcoded_jailbreaks
        input_labels = tc['toxicity'] + non_jailbreak_toxic_tc['toxicity'] + [1 for i in hardcoded_jailbreaks]

    else:
        inputs = tc['user_input']
        input_labels = tc['toxicity']

    tc_df = pd.DataFrame({'user_input':inputs, 'toxicity':input_labels})

    print("Exporting")
    tc_df.to_csv(f'data/input_dataset_{split}.csv')


if __name__ == '__main__':
    """
    Worth considering is we might not want to unlearn jailbreaks so that we can see if unlearning mitigates them on its own without them.
    """

    main('train', jailbreak=False)
    main('test', jailbreak=True)



