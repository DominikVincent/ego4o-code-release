from langchain_ollama import OllamaLLM
import json



llm = OllamaLLM(model="llama3.1")


result_path = '/CT/EgoMocap/work/LLaVA/eval_out/test_nymeria_all10_29_00_12_29/result.json'

with open(result_path) as f:
    data_list = json.load(f)

# make the prompt input

prompt_str = 'I will provide several sentences describing human motion. Please follow the style of the reference sentences provided and rephrase the text I will share with you. Please only respnse with the modified text. Do not include the original text in your response.'

sample_gt_text = [data_item['gt_text'] for data_item in data_list[::10]]
prompt_input = prompt_str + '\n\nThis is the reference sentences:' + '\n\n'.join(sample_gt_text) + '\n\nThis is the text to be re-phrased:' + '\n\n'.join(data_list[0]['pred_text'])

response = llm.invoke(prompt_input)
print(response)
for i in range(1, len(data_list)):
    pred_sample_text = data_list[i]['pred_text']

    prompt_input = 'This is the text to be re-phrased:' + '\n\n'.join(pred_sample_text)

    response = llm.invoke(prompt_input)
    print(response)
