from sentence_transformers import SentenceTransformer
import pickle
from torch.utils.data import Dataset
import os
from util.globals import *
from dsets import CounterFactDataset


def encode_ike_facts(sentence_model: SentenceTransformer, ds: Dataset):

    sentences = []
    for i, train_data in enumerate(ds):
        rewrite = train_data['requested_rewrite']
        subject = rewrite['subject']
        target_new = rewrite['target_new']['str']
        target_true = rewrite['target_true']['str']

        new_fact = rewrite['prompt'].format(subject) + ' ' + target_new

        paraphrases = train_data['paraphrase_prompts'][0]
        neighbors = train_data['neighborhood_prompts'][0]

        sentences.append(f"New Fact: {new_fact}\nPrompt: {new_fact}\n\n")
        sentences.append(
            f"New Fact: {new_fact}\nPrompt: {paraphrases} {target_new}\n\n")
        sentences.append(
            f"New Fact: {new_fact}\nPrompt: {neighbors} {target_true}\n\n")

    embeddings = sentence_model.encode(sentences)
    base_path = EMBEDDING_DIR
    os.makedirs(base_path, exist_ok=True)
    safe_model_name = 'all-MiniLM-L6-v2'
    with open(f'{base_path}/{safe_model_name}_{type(ds).__name__}.pkl', "wb") as fOut:
        pickle.dump({'sentences': sentences, 'embeddings': embeddings}, fOut,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print("Demonstrations Encoded!")


# def encode_ike_facts(sentence_model: SentenceTransformer, ds: Dataset):
#     case_data = {
#         "direct": {"sentences": [], "embeddings": []},
#         "paraphrase": {"sentences": [], "embeddings": []},
#         "locality": {"sentences": [], "embeddings": []}
#     }

#     for i, train_data in enumerate(ds):
#         rewrite = train_data['requested_rewrite']

#         subject = rewrite['subject']
#         target_new = rewrite['target_new']['str']
#         target_true = rewrite['target_true']['str']

#         new_fact = rewrite['prompt'].format(subject) + ' ' + target_new

#         paraphrases = train_data['paraphrase_prompts'][0]
#         neighbors = train_data['neighborhood_prompts'][0]

#         cases = {
#             "direct": f"New Fact: {new_fact}\nPrompt: {new_fact}\n\n",
#             "paraphrase": f"New Fact: {new_fact}\nPrompt: {paraphrases} {target_new}\n\n",
#             "locality": f"New Fact: {new_fact}\nPrompt: {neighbors} {target_true}\n\n"
#         }
#         for k, v in cases.items():
#             case_data[k]["sentences"].append(v)

#     # 批量计算每种情况的嵌入
#     for case in case_data:
#         case_data[case]["embeddings"] = sentence_model.encode(
#             case_data[case]["sentences"])

#     # 存储
#     base_path = EMBEDDING_DIR
#     os.makedirs(base_path, exist_ok=True)
#     safe_model_name = 'all-MiniLM-L6-v2'
#     with open(f'{base_path}/{safe_model_name}_{type(ds).__name__}.pkl', "wb") as fOut:
#         pickle.dump(case_data, fOut, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    print("Instantializing Model...")

    model = SentenceTransformer('/data1/yexiaotian/models/all-MiniLM-L6-v2')
    print(f"DATA_DIR={DATA_DIR}")

    ds = CounterFactDataset(DATA_DIR, training_size=10000)

    encode_ike_facts(
        sentence_model=model,
        ds=ds,
    )
