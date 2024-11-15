from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
import pandas as pd
import googletrans, random
from tqdm import tqdm
import bitsandbytes as bnb
import re
import json

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
local_model = "C:\\somethings\\checkpoint-150-merged"



# 用修改后的配置加载模型


prompt = """
You are a specialized intelligent assistant for screening participants for clinical trials, and you possess extensive medical background knowledge. 
Your task is to process and summarize the patient's case based on the patient information provided. This information will be used to match the inclusion/exclusion criteria of the clinical trial in the future.
These criteria typically include information such as age, gender, type and stage of disease, and previous treatment history. 
Regardless of the language of your received, please output in English only. 

Please follow the steps below:
1. Read all the patient information provided.
2. If the patient information is less than 110k tokens, then for each summary, you should use brackets to supplement the original part.
The format is: <your summary>(<corresponding original text>)。 
3. If the patient information is more than 110k tokens, then make an appropriate summary of the patient information. For each summary, you should use brackets to supplement only the most critical part of the original text. The format is: <your summary>(...<keywords>...)

——————————
Now take a deep breath, and think step by step and complete the task. Remember to output in English only.
"""
# 患者病历输入，JSON格式（目前还是txt）
with open('C:/somethings/thesisB_0809/testOutput1013.txt', mode='r', encoding="utf-8") as file:
    input = file.read().replace('\n', '')

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True  
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=quantization_config,
)

messages = [
    {"role": "system", "content": prompt},
    # {"role": "system", "content": 病例},
    {"role": "user", "content": input},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True, # Add a generation prompt to the input
    return_tensors="pt"     # Return PyTorch tensors
).to(model.device)

# 仅需要一次
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

"""
生成参数调节
input_ids：模型的输入。
max_new_tokens：最大生成的 token 数，控制输出长度。(输入输出总计8192tokens)
eos_token_id：结束符，控制生成何时结束。
do_sample：是否启用采样,False 为贪婪搜索。
Temperature：0-1, 温度越低文本随机性越低。0为贪婪搜索。
top_p：0-1, 采样概率阈值，指定了模型从其概率分布中选取生成 token 的范围。
"""
print("begin part1")
outputs = model.generate(
    input_ids,
    max_new_tokens=2048,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.2, 
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
#输出结果，并解码保存
generated_text = tokenizer.decode(response, skip_special_tokens=True)
print("part1:")
print(generated_text)


# 第二部分，根据生成的总结，进行病种判断。

part2_prompt = """
Please judge what kind of disease the patient has based on his/her case. 
The diseases include: cancer, cardiovascular disease, diabetes, nervous system disease, infectious disease. If none of them match, then it means it does not match. And give your own judgment.
Strictly follow the following output format: <disease type>. The specific disease you think it is.
Remember to output in English only.
"""


# 构建新的对话消息列表，包含疾病判断任务
part2_messages = [
    {"role": "system", "content": part2_prompt},
    {"role": "user", "content": generated_text},
]

# 将消息转化为模型的输入格式
part2_input_ids = tokenizer.apply_chat_template(
    part2_messages,
    add_generation_prompt=True,  # Add a generation prompt to the input
    return_tensors="pt"  # 返回PyTorch tensors
).to(model.device)


# 生成疾病判断的结果
part2_outputs = model.generate(
    part2_input_ids,
    max_new_tokens=100,  # 可以根据需要调整输出token长度
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.2,
    top_p=0.9,
)

# 解码输出，并保存或显示疾病判断结果
part2_response = part2_outputs[0][part2_input_ids.shape[-1]:]
part2_generated_text = tokenizer.decode(part2_response, skip_special_tokens=True)
print("part2:")
print(part2_generated_text, type(part2_generated_text))

# #第三部分,匹配临床试验
# translate to eng
translator = googletrans.Translator()
p2_generated_text = part2_generated_text

# 疾病种类库
disease_dict = {
    "cancer": "癌",
    "cardiovascular disease": "cardiovascular disease",
    "diabetes": "diabetes",
    "nervous system disease": "nervous system disease",
    "infectious disease": "infectious disease"
}
print('begin part3')
p3_result = []
for key, value in disease_dict.items():
    if key in p2_generated_text.lower():
        disease = disease_dict[key]
        stopRunning = False
        break
    if value in p2_generated_text:
        disease = key
        stopRunning = False
        break
else:
    disease = f"This disease is not in our common matching library. The following is our reference for judgment: {part2_generated_text}"
    stopRunning = True

if stopRunning == False:
    cinicalTrialPath = f'C:/somethings/thesisB_0809/clinicalTrial/{disease}.csv'
    df = pd.read_csv(cinicalTrialPath)

    inclusion_criteria = []
    exclusion_criteria = []

    for i in tqdm(range(5)):
        # 读取当前行的排入和排出标准
        index = random.randint(0, len(df) - 1)
        inclusion_criteria.append(df.iloc[index, 2])  # 第三列是排入标准
        exclusion_criteria.append(df.iloc[index, 3])  # 第四列是排出标准

        #这里可以打印当前组的排入标准和排出标准了


    part3Prompt = """
    You are a specialized intelligent assistant for screening participants for clinical trials. All patients tested were considered to have consented to participate in the trial.

    Please strictly follow the format below when generating the output:

    1. Identify all inclusion criteria and list them:

    - After each inclusion criterion, specify its relation to the patient case (Eligible, Excluded, Irrelevant, or Lacking Information).
    - Total number of inclusion criteria.
    - Total number of eligible inlucsion criteria.
    - Total number of excluded inlucsion criteria.
    - Total number of irrelevant inlucsion criteria.
    - Total number of lacking information inlucsion criteria.

    2. Identify all exclusion criteria and list them:
    - After each exclusion criterion, specify its relation to the patient case (Eligible, Excluded, Irrelevant, or Lacking Information).
    - Total number of exclusion criteria.
    - Total number of eligible exlucsion criteria.
    - Total number of excluded exlucsion criteria.
    - Total number of irrelevant exlucsion criteria.
    - Total number of lacking information exlucsion criteria.
    ———————— 
    Take a deep breath, strictly follow the above format, and proceed step-by-step to complete the task.
    """


    for i in tqdm(range(5)):
        part3_messages = [
            {"role": "system", "content": part3Prompt},
            {"role": "user", "content": f"The patient has {disease}. The inclusion criteria is {inclusion_criteria[i]}, and the exclusion criteria is {exclusion_criteria[i]}."},
        ]

        # 将消息转化为模型的输入格式
        part3_input_ids = tokenizer.apply_chat_template(
            part3_messages,
            add_generation_prompt=True,  # Add a generation prompt to the input
            return_tensors="pt"  # 返回PyTorch tensors
        ).to(model.device)

        # print("begin part3")
        # 生成疾病判断的结果
        print(f"begin part3 {i}")
        part3_outputs = model.generate(
            part3_input_ids,
            max_new_tokens=2048,  # 可以根据需要调整输出token长度
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
        )

        # 解码输出，并保存或显示疾病判断结果
        part3_response = part3_outputs[0][part3_input_ids.shape[-1]:]
        part3_generated_text = tokenizer.decode(part3_response, skip_special_tokens=True)
        # print("part3:")
        # print(part3_generated_text)
        

        part4Messages = part3_generated_text

        part4Prompt = """
        The input in this section pertains to the matching between patient information and clinical trial criteria. You must output the dictionary strictly in the following format:

        {
        "Inclusion Criteria": {
            "total number of inclusion": "<Total number of inclusion criteria>",
            "eligible inclusion": "<Number of inclusion criteria met>",
            "excluded inclusion": "<Number of inclusion criteria not met>",
            "irrelevant inclusion": "<Number of irrelevant inclusion criteria>",
            "lacking information inclusion": "<Number of inclusion criteria lacking information>"
        },
        "Exclusion Criteria": {
            "total number of exclusion": "<Total number of exclusion criteria>",
            "eligible exclusion": "<Number of exclusion criteria met>",
            "excluded exclusion": "<Number of exclusion criteria not met>",
            "irrelevant exclusion": "<Number of irrelevant exclusion criteria>",
            "lacking information exclusion": "<Number of exclusion criteria lacking information>"
        }
        }

        """


        part4Messages = [
            {"role": "system", "content": part4Prompt},
            {"role": "user", "content": part4Messages},
        ]

        # 将消息转化为模型的输入格式
        part4_input_ids = tokenizer.apply_chat_template(
            part4Messages,
            add_generation_prompt=True,  # Add a generation prompt to the input
            return_tensors="pt"  # 返回PyTorch tensors
        ).to(model.device)

        # 生成疾病判断的结果
        part4_outputs = model.generate(
            part4_input_ids,
            max_new_tokens=100,  # 可以根据需要调整输出token长度
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
        )

        # 解码输出，并保存或显示疾病判断结果
        part4_response = part4_outputs[0][part4_input_ids.shape[-1]:]
        part4_generated_text = tokenizer.decode(part4_response, skip_special_tokens=True)
        print("part4:")
        print(part4_generated_text, type(part4_generated_text))


        # 使用正则提取数据
        matches = re.search(r'{.*}', part4_generated_text, re.DOTALL)

        json_data_str = matches.group()  # 提取外层的大括号内容

        # 将其转换为字典对象
        json_data = json.loads(json_data_str)

        # 第二次分别处理每个嵌套的大括号
        inclusion_criteria = json_data.get("Inclusion Criteria", {})
        exclusion_criteria = json_data.get("Exclusion Criteria", {})

        # 提取下述 10 个参数并赋值给变量
        totalInclusion = inclusion_criteria.get("total number of inclusion")
        eligibleInclusion = inclusion_criteria.get("eligible inclusion")
        excludedInclusion = inclusion_criteria.get("excluded inclusion")
        irrelevantInclusion = inclusion_criteria.get("irrelevant inclusion")
        lackingInfoInclusion = inclusion_criteria.get("lacking information inclusion")

        totalExclusion = exclusion_criteria.get("total number of exclusion")
        eligibleExclusion = exclusion_criteria.get("eligible exclusion")
        excludedExclusion = exclusion_criteria.get("excluded exclusion")
        irrelevantExclusion = exclusion_criteria.get("irrelevant exclusion")
        lackingInfoExclusion = exclusion_criteria.get("lacking information exclusion")

        # 排除分数
        exclusionScore = int(eligibleExclusion)

        # 等级分数
        inclusionPercentage = int(eligibleInclusion) / int(totalInclusion)
        exclusionPercentage = int(eligibleExclusion) / int(totalExclusion) if int(totalExclusion) > 0 else 0
        levelScore = (inclusionPercentage - exclusionPercentage)*100

        print(f"The exlusionScore is: {exclusionScore}, and the levelScore is: {levelScore}")
        with open(f'C:/somethings/thesisB_0809/testOutput{i}.txt', mode='w', encoding="utf-8") as file:
            file.write(part3_generated_text)
            file.write("part4:")
            file.write(part4_generated_text)
            file.write("\n")
            file.write(f"The exlusionScore is: {exclusionScore}, and the levelScore is: {levelScore}")

else:
    print(disease)
        


# Part 4, 等级与分数

    


