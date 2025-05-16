import pandas as pd
import os
from PIL import Image
import io
import json
from datasets import Dataset, Sequence
from typing import List, Optional, Tuple,Union

import base64
from io import BytesIO
from PIL import Image
import pandas as pd
import numpy as np
import random
from tqdm.auto import tqdm
from multiprocessing import cpu_count
from functools import lru_cache

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

n1 = "The message makes me more concerned about the health risks of poor eating habits - Scale: 1 (not at all) - 9 (extremely)"
n2 = "The message motivates me to make healthy eating choices - Scale: 1 (not at all) - 9 (extremely)"
n3 = "In your opinion, how harmful is neglecting proper nutrition and weight management to your overall health? - Scale: 0 (not at all)-6 (extremely harmful)"
n4 = "How open are you to adopting healthier eating habits and lifestyle changes? - Scale: 1 (not at all)-9 (extremely)"
n1_1 = "The message makes me more concerned about the health risks of poor eating habits - Scale: 1 (not at all) - 9 (extremely)"
n1_2 = "The message makes me more concerned about the health risks of diabetes. - Scale: 1 (not at all) - 9 (extremely)"
n2_2 = "The message motivates me to maintain proper blood sugar control and a healthy lifestyle. - Scale: 1 (not at all) - 9 (extremely)"
n3_1 = "In your opinion, how harmful is neglecting diabetes management to your general health? - Scale: 0 (not at all)-6 (extremely harmful)"
n4_1 = "How open are you to maintaining healthy diabetes care practices in the future? - Scale: 1 (not at all)-9 (extremely)"
n2_3 = "The message motivates me to maintain proper blood sugar control and a healthy lifestyle - Scale: 1 (not at all) - 9 (extremely)"

#Mental Health: 
mh1 = "The message makes me more concerned about importance of mental health - Scale: 1 (not at all) - 9 (extremely)"
mh2 = "The message motivates me to prioritize my mental well-being and seek support when needed. - Scale: 1 (not at all) - 9 (extremely)"
mh3 = "In your opinion, how harmful is ignoring mental health to your overall quality of life? - Scale: 0 (not at all)-6 (extremely harmful)"
mh4 = "How open are you to adopting practices that promote good mental health in the future? - Scale: 1 (not at all)-9 (extremely)"
mh1_1 = "The message makes me more concerned about the importance of mental health - Scale: 1 (not at all) - 9 (extremely)"
mh3_1 = "In your opinion, how harmful is neglecting mental health to your overall quality of life? - Scale: 0 (not at all)-6 (extremely harmful)"

#Sexual Practice:
sp_1 = "The message makes me more concerned about the health risks of unsafe sexual practices - Scale: 1 (not at all) - 9 (extremely)"
sp_2 = "The message motivates me to not engage in unsafe sexual behavior - Scale: 1 (not at all) - 9 (extremely)"
sp_3 = "In your opinion, how harmful is unsafe sexual behavior to your general health? - Scale: 0 (not at all)-6 (extremely harmful)"
sp_4 = "How open are you to practicing safe sex in the future? - Scale: 1 (not at all)-9 (extremely)"
sp_1_1 = "The message makes me more concerned about the health risks of unsafe sexual practices. - Scale: 1 (not at all) - 9 (extremely)"
sp2_2 = "The message motivates me to  not to engage in unsafe sexual behavior - Scale: 1 (not at all) - 9 (extremely)"
sp2_3 = "The message motivates me to  not engage in unsafe sexual behavior - Scale: 1 (not at all) - 9 (extremely)"
sp2_4 = "The message motivates me to to not engage in unsafe sexual behavior. - Scale: 1 (not at all) - 9 (extremely)"
#Substance Abuse:

sa1 = "The message makes me more concerned about the health risks of substance abuse. - Scale: 1 (not at all) - 9 (extremely)"
sa1_1 = "The message makes me more concerned about the health risks of substance abuse - Scale: 1 (not at all) - 9 (extremely)"
sa_2 = "The message motivates me to not use substances. - Scale: 1 (not at all) - 9 (extremely)"
sa_3 = "In your opinion, how harmful is substance use to your general health? - Scale: 0 (not at all)-6 (extremely harmful)"
sa_4 = "How open are you to trying a substance in the future? - Scale: 1 (not at all)-9 (extremely)"

#Smoking:
s_tg1 = "The message makes me more concerned about the health risks of COPD and smoking. - Scale: 1 (not at all) - 9 (extremely)"
s_tg1_1 = "The message makes me more concerned about the health risks of COPD and smoking - Scale: 1 (not at all) - 9 (extremely)"
s_tg2 = "The message motivates me to not smoke. - Scale: 1 (not at all) - 9 (extremely)"
s_tg2_2 = "The message motivates me to not to smoke. - Scale: 1 (not at all) - 9 (extremely)"
s_tg3 = "In your opinion, how harmful is smoking to your general health? - Scale: 0 (not at all)-6 (extremely harmful)"
s_tg4 = "How open are you to smoking in the future? - Scale: 1 (not at all)-9 (extremely)"
    
#Chronic Disease: 
tg0 = "The message makes me more concerned about the health risks of of untreated arthritis. - Scale: 1 (not at all) - 9 (extremely)"
tg0_1= "The message makes me more concerned about the health risks of untreated arthritis. - Scale: 1 (not at all) - 9 (extremely)"
tg1 = "The message motivates me to not ignore arthritis symptoms. - Scale: 1 (not at all) - 9 (extremely)"
tg2 = "In your opinion, how harmful is leaving arthritis untreated to your general health? - Scale: 0 (not at all)-6 (extremely harmful)"
tg3 = "How open are you to managing arthritis through recommended treatments in the future? - Scale: 1 (not at all)-9 (extremely)"
tg0_2 = "The message makes me more concerned about the health risks of heart disease. - Scale: 1 (not at all) - 9 (extremely)"
tg1_1 = "The message motivates me to not ignore cardiovascular health - Scale: 1 (not at all) - 9 (extremely)"
tg2_1 = "In your opinion, how harmful is ignoring heart health to your general health? - Scale: 0 (not at all)-6 (extremely harmful)"
tg3_1 = "How open are you to maintaining heart-healthy practices in the future? - Scale: 1 (not at all)-9 (extremely)"
tg0_3 = "The message makes me more concerned about the health risks of cystic fibrosis - Scale: 1 (not at all) - 9 (extremely)"
tg1_2 = "The message motivates me to not ignore cystic fibrosis symptoms - Scale: 1 (not at all) - 9 (extremely)"
tg2_2 = "In your opinion, how harmful is ignoring cystic fibrosis symptoms to your general health? - Scale: 0 (not at all)-6 (extremely harmful)"
tg3_2 = "How open are you to maintaining awareness about your respiratory health conditions in the future? - Scale: 1 (not at all)-9 (extremely)"
tg1_3 = "The message motivates me not to ignore cystic fibrosis symptoms - Scale: 1 (not at all) - 9 (extremely)"
tg3_3 = "How open are you to maintaining awareness about your respiratory health conditions in the future? - Scale: 1 (not at all)-9 (extremely open)"

#Vaccination: 
tg1_v = "The message makes me more concerned about the health risks of skipping vaccinations. - Scale: 1 (not at all) - 9 (extremely)"
tg2_v = "The message motivates me to not skip recommended vaccinations. - Scale: 1 (not at all) - 9 (extremely)"
tg3_v = "In your opinion, how harmful is skipping vaccinations to your general health? - Scale: 0 (not at all)-6 (extremely harmful)"
tg4_v = "How open are you to getting recommended vaccinations in the future? - Scale: 1 (not at all)-9 (extremely)"

hiv1 = "The message makes me more concerned about the health risks of HIV/AIDS - Scale: 1 (not at all) - 9 (extremely)"
hiv2 = "The message motivates me to practice safe behaviors. - Scale: 1 (not at all) - 9 (extremely)"
hiv3 = "In your opinion, how harmful is neglecting HIV treatment and prevention to your general health? - Scale: 0 (not at all)-6 (extremely harmful)"
hiv4 = "How open are you to maintaining consistent HIV care and prevention practices in the future? - Scale: 1 (not at all)-9 (extremely)"
hiv5 = "The message motivates me to practice safe behaviors. - Scale: 1 (not at all) - 9 (extremely)"
hiv6 = "How open are you to maintaining consistent HIV/aids care and prevention practices in the future? - Scale: 1 (not at all)-9 (extremely)"

shared1 = "To what extent did the material make you feel sad? - Scale: 1 (not at all) -9 (extremely)"
shared2= "To what extent did the material make you feel angry? - Scale: 1 (not at all) -9 (extremely)"
shared3 = "To what extent did the material make you feel afraid? - Scale: 1 (not at all) -9 (extremely)"
shared4 = "To what extent did the material make you feel guilty? - Scale: 1 (not at all) -9 (extremely)"
shared5 = "To what extent did the material make you feel disgusted? - Scale: 1 (not at all) -9 (extremely)"
shared6 = "To what extent did the material make you feel worried? - Scale: 1 (not at all) -9 (extremely)"
shared7 = "To what extent did the material make you feel ashamed? - Scale: 1 (not at all) -9 (extremely)"
shared8 = "To what extent did the material make you feel hopeful? - Scale: 1 (not at all) -9 (extremely)"    
   
questions_by_topic = {
    'Nutrition': [n1, n2, n3, n4,shared1, shared2, shared3, shared4, shared5, shared6, shared7, shared8],
    'Diabetes': [n1_2, n2_2, n3_1, n4_1,shared1, shared2, shared3, shared4, shared5, shared6, shared7, shared8],
    'Mental Health': [mh1, mh2, mh3, mh4,shared1, shared2, shared3, shared4, shared5, shared6, shared7, shared8],
    'Sexual Health': [sp_1, sp_2, sp_3, sp_4,shared1, shared2, shared3, shared4, shared5, shared6, shared7, shared8],
    'Substance abuse': [sa1, sa_2, sa_3, sa_4,shared1, shared2, shared3, shared4, shared5, shared6, shared7, shared8],
    'Smoking and COPD': [s_tg1, s_tg2, s_tg3, s_tg4,shared1, shared2, shared3, shared4, shared5, shared6, shared7, shared8],
    'Chronic Diseases': [tg0_1, tg1, tg2, tg3,shared1, shared2, shared3, shared4, shared5, shared6, shared7, shared8],
    'Vaccination': [tg1_v, tg2_v, tg3_v, tg4_v,shared1, shared2, shared3, shared4, shared5, shared6, shared7, shared8],
    'HIV': [hiv1, hiv2, hiv3, hiv4,shared1, shared2, shared3, shared4, shared5, shared6, shared7, shared8]
}
     
def get_mc_questions(row):
    topic_name = row['topic']
    if topic_name == 'Nutrition':
        return [n1, n2, n3, n4, shared1, shared2, shared3, shared4, shared5, shared6, shared7, shared8]
    elif topic_name == 'Vaccination':
        return [tg1_v, tg2_v, tg3_v, tg4_v, shared1, shared2, shared3, shared4, shared5, shared6, shared7, shared8]
    elif topic_name == 'Mental Health':
        return [mh1, mh2, mh3, mh4, shared1, shared2, shared3, shared4, shared5, shared6, shared7, shared8]
    elif topic_name == 'Substance abuse':
        return [sa1, sa_2, sa_3, sa_4, shared1, shared2, shared3, shared4, shared5, shared6, shared7, shared8]
    elif topic_name == 'Sexual Health':
        return [sp_1, sp_2, sp_3, sp_4, shared1, shared2, shared3, shared4, shared5, shared6, shared7, shared8]
    elif topic_name == 'HIV/aids':
        return [hiv1, hiv2, hiv3, hiv4, shared1, shared2, shared3, shared4, shared5, shared6, shared7, shared8]
    elif topic_name == 'Smoking and COPD':
        return [s_tg1, s_tg2, s_tg3, s_tg4, shared1, shared2, shared3, shared4, shared5, shared6, shared7, shared8]
    elif topic_name == 'Chronic Diseases':
        return [tg0_1, tg1, tg2, tg3, shared1, shared2, shared3, shared4, shared5, shared6, shared7, shared8]
    else:
        return []
    
def get_mc_questions(row):
    topic = row['topic']
    topic_questions = questions_by_topic[topic]
    target_question = row['question']
 
    topic_questions = questions_by_topic.get(topic, [])
    prompt_questions = [q for q in topic_questions if q.strip() != target_question.strip()]

    return prompt_questions

def get_baseline_questions(row):
    topic_name = row['topic']
    topic_map = {
        'Nutrition': row['nutrition'],
        'Vaccination': row['vaccination'],
        'Mental Health': row['mental_health'],
        'Substance abuse': row['substance_abuse'],
        'Sexual Health': row['sexual_health'],
        'HIV': row['hiv'],
        'Smoking and COPD': row['copd'],
        'Chronic Diseases': row['chronic_disease']
    }
    prompts = {
        'Nutrition': [
            "I am concerned about the health risks related to eating unhealthy (0-9): {}",
            "I am motivated to make healthy eating choices (0-9): {}",
            "Eating unhealthy is harmful to my health (0-9): {}",
            "I am open to eating healthy in the future (0-9): {}"
        ],
        'Vaccination': [
            "I am concerned about the health risks of skipping recommended vaccinations (0-9): {}",
            "I am motivated to not skip recommended vaccinations (0-9): {}",
            "Skipping recommended vaccinations is harmful to my health (0-9): {}",
            "I am open to getting recommended vaccinations in the future (0-9): {}"
        ],
        'Mental Health': [
            "I am concerned about the health risks related to mental health (0-9): {}",
            "I am motivated to prioritize my mental well-being (0-9): {}",
            "Ignoring mental health is harmful to my health (0-9): {}",
            "I am open to adopting practices that promote good mental health in the future (0-9): {}"
        ],
        'Substance abuse': [
            "I am concerned about the health risks of using substance abuse (0-9): {}",
            "I am motivated to not use substances (0-9): {}",
            "Substance abuse is harmful to my health (0-9): {}",
            "I am open to trying a substance in the future (0-9): {}"
        ],
        'Sexual Health': [
            "I am concerned about the health risks of unsafe sexual practices (0-9): {}",
            "I am motivated to not engage in unsafe sexual practices (0-9): {}",
            "Unsafe sexual behavior is harmful to my health (0-9): {}",
            "I am open to practicing safe sex in the future (0-9): {}"
        ],
        'HIV': [
            "I am concerned about the health risks of HIV/aids (0-9): {}",
            "I am motivated to practice safe behaviors (0-9): {}",
            "Neglecting HIV treatments and prevention is harmful to my health (0-9): {}",
            "I am open to maintaining consistent HIV care and prevention in the future (0-9): {}"
        ],
        'Smoking and COPD': [
            "I am concerned about the health risks of COPD and smoking (0-9): {}",
            "I am motivated to not smoke (0-9): {}",
            "Smoking is harmful to my health (0-9): {}",
            "I am open to smoking in the future (0-9): {}"
        ],
        'Chronic Diseases': [
            "I am concerned about the health risks of chronic diseases (0-9): {}",
            "I am motivated to not ignore chronic disease symptoms (0-9): {}",
            "Ignoring chronic disease symptoms is harmful to my health (0-9): {}",
            "I am open to regular health screenings and preventive care in the future (0-9): {}"
        ]
    }

    if topic_name not in topic_map:
        return []

    responses = topic_map[topic_name]
    questions = [
        prompts[topic_name][0].format(responses[2]),
        prompts[topic_name][1].format(responses[7]),
        prompts[topic_name][2].format(responses[12]),
        prompts[topic_name][3].format(responses[17]),
    ]
    random.shuffle(questions)
    return questions


def get_demographics(row):
    
    demographics = {
        "Age": row['age'],
        "Gender": row['gender'],
        "Religion": row['religion'],
        "Political Affiliation": row['political_aff'] if row['political_aff'] else row['political_aff_text'],
        "Race/Ethnicity": row['race_ethnicity'],
        "Language": row['prim_language'],
        "Employment": row['employment_status'] if row['employment_status'] else row['employment_status_text'],
        "Education": row['highest_education'],
        "Profession": row['current_profession'] if row['current_profession'] else row['current_profession_text'],
        "Income": row['income'],
        "Disability": row['conditional_disability'],
        "Marital Status": row['marital_status'] if row['marital_status'] else row['marital_status_text'],
        "Family Status": row['family_status'] if row['family_status'] else row['family_status_text'],
        "Disability": row['conditional_disability'] if row['conditional_disability'] else row['disability_binary'] if row['disability_binary'] else 'None',
    }

    num_to_select = np.random.randint(1, len(demographics) + 1)
    selected_keys = np.random.choice(list(demographics.keys()), size=num_to_select, replace=False)

    return ', '.join(f"{key}: {demographics[key]}" for key in selected_keys)


def get_personality(row):
    personality = {
        "Trust": row['Trust'],
        "Depression": row['Depression'],
        "Productiveness": row['Productiveness'],
        "Assertiveness": row['Assertiveness'],
        "Extraversion": row['Extraversion'],
        "Open-Mindedness": row['Open-Mindedness'],
        "Organization": row['Organization'],
        "Anxiety": row['Anxiety'],
        "Intellectual Curiosity": row['Intellectual Curiosity'],
        "Energy Level": row['Energy Level'],
        "Aesthetic Sensitivity": row['Aesthetic Sensitivity'],
        "Conscientiousness": row['Conscientiousness'],
        "Agreeableness": row['Agreeableness'],
        "Compassion": row['Compassion'],
        "Emotional Volatility": row['Emotional Volatility'],
        "Creative Imagination": row['Creative Imagination'],
        "Sociability": row['Sociability'],
        "Respectfulness": row['Respectfulness'],
        "Negative Emotionality": row['Negative Emotionality'],
        "Responsibility": row['Responsibility']
    }
    
    num_to_select = np.random.randint(1, len(personality) + 1)
    selected_keys = np.random.choice(list(personality.keys()), size=num_to_select, replace=False)
    return ', '.join(f"{key} (1-5): {personality[key]}" for key in selected_keys)
    

def get_baseline(row):
    topic_questions = get_baseline_questions(row)
    
    if topic_questions:
        return '\n'.join([f"Baseline {i+1}: {q}" for i, q in enumerate(topic_questions)])
    else:
        return "No baseline questions available for this topic."

def get_curr_question(question):
    if question.endswith('_angry'):
        q = "To what extent did the material make you feel angry? - Scale: 1 (not at all) -9 (extremely)"
    elif question.endswith('_sad'):
        q = "To what extent did the material make you feel sad? - Scale: 1 (not at all) -9 (extremely)"
    elif question.endswith('_afraid'):
        q = "To what extent did the material make you feel afraid? - Scale: 1 (not at all) -9 (extremely)"
    elif question.endswith('_guilty'):
        q = "To what extent did the material make you feel guilty? - Scale: 1 (not at all) -9 (extremely)"
    elif question.endswith('_disgusted') or question.endswith('_digusted'):
        q = "To what extent did the material make you feel disgusted? - Scale: 1 (not at all) -9 (extremely)"
    elif question.endswith('_worried'):
        q = "To what extent did the material make you feel worried? - Scale: 1 (not at all) -9 (extremely)"
    elif question.endswith('_ashamed'):
        q = "To what extent did the material make you feel ashamed? - Scale: 1 (not at all) -9 (extremely)"
    elif question.endswith('_hopeful'):
        q = "To what extent did the material make you feel hopeful? - Scale: 1 (not at all) -9 (extremely)"
    elif question.endswith('_free_form'):
        q = "Type in every thought that came to mind viewing this material."
    elif question == 'hiv_mc1':
        q = "The message makes me more concerned about the health risks of HIV/AIDS - Scale: 1 (not at all) - 9 (extremely)"
    elif question == 'hiv_mc2':
        q = "The message motivates me to practice safe behaviors - Scale: 1 (not at all) - 9 (extremely)"
    elif question.endswith('hiv_mc3'):
        q = "In your opinion, how harmful is neglecting HIV treatment and prevention to your general health? - Scale: 0 (not at all)-6 (extremely harmful)"
    elif question.endswith('hiv_mc4'):
        q = "How open are you to maintaining consistent HIV care and prevention practices in the future? - Scale: 1 (not at all)-9 (extremely)"
    elif question.endswith('chronic_disease_mc1'):
        q = "The message makes me more concerned about the health risks of untreated arthritis. - Scale: 1 (not at all) - 9 (extremely)"
    elif question.endswith('chronic_disease_mc2'):
        q = "The message motivates me to not ignore arthritis symptoms. - Scale: 1 (not at all) - 9 (extremely)"
    elif question.endswith('chronic_disease_mc3'):
        q = "In your opinion, how harmful is leaving arthritis untreated to your general health? - Scale: 0 (not at all)-6 (extremely harmful)"
    elif question.endswith('chronic_disease_mc4'):
        q = "How open are you to managing arthritis through recommended treatments in the future? - Scale: 1 (not at all)-9 (extremely)"
    elif question.endswith('nutrition_mc1'):
        q = "The message makes me more concerned about the health risks of poor eating habits - Scale: 1 (not at all) - 9 (extremely)"
    elif question.endswith('nutrition_mc2'):
        q = "The message motivates me to make healthy eating choices - Scale: 1 (not at all) - 9 (extremely)"
    elif question.endswith('nutrition_mc3'):
        q = "In your opinion, how harmful is neglecting proper nutrition and weight management to your overall health? - Scale: 0 (not at all)-6 (extremely harmful)"
    elif question.endswith('nutrition_mc4'):
        q = "How open are you to adopting healthier eating habits and lifestyle changes? - Scale: 1 (not at all)-9 (extremely)"
    elif question.endswith('vaccination_mc1'):
        q = "The message makes me more concerned about the health risks of skipping vaccinations. - Scale: 1 (not at all) - 9 (extremely)"
    elif question.endswith('vaccination_mc2'):
        q = "The message motivates me to not skip recommended vaccinations. - Scale: 1 (not at all) - 9 (extremely)"
    elif question.endswith('vaccination_mc3'):
        q = "In your opinion, how harmful is skipping vaccinations to your general health? - Scale: 0 (not at all)-6 (extremely harmful)"
    elif question.endswith('vaccination_mc4'):
        q = "How open are you to getting recommended vaccinations in the future? - Scale: 1 (not at all)-9 (extremely)"
    elif question.endswith('mental_health_mc1'):
        q = "The message makes me more concerned about the importance of mental health - Scale: 1 (not at all) - 9 (extremely)"
    elif question.endswith('mental_health_mc2'):
        q = "The message motivates me to prioritize my mental well-being and seek support when needed. - Scale: 1 (not at all) - 9 (extremely)"
    elif question.endswith('mental_health_mc3'):
        q = "In your opinion, how harmful is ignoring mental health to your overall quality of life? - Scale: 0 (not at all)-6 (extremely harmful)"
    elif question.endswith('mental_health_mc4'):
        q = "How open are you to adopting practices that promote good mental health in the future? - Scale: 1 (not at all)-9 (extremely)"
    elif question.endswith('substance_abuse_mc1'):
        q = "The message makes me more concerned about the health risks of substance abuse - Scale: 1 (not at all) - 9 (extremely)"
    elif question.endswith('substance_abuse_mc2'):
        q = "The message motivates me to not use substances. - Scale: 1 (not at all) - 9 (extremely)"
    elif question.endswith('substance_abuse_mc3'):
        q = "In your opinion, how harmful is substance use to your general health? - Scale: 0 (not at all)-6 (extremely harmful)"
    elif question.endswith('substance_abuse_mc4'):
        q = "How open are you to trying a substance in the future? - Scale: 1 (not at all)-9 (extremely)"
    elif question.endswith('smoking_mc1'):
        q = "The message makes me more concerned about the health risks of COPD and smoking - Scale: 1 (not at all) - 9 (extremely)"
    elif question.endswith('smoking_mc2'):
        q = "The message motivates me to not smoke. - Scale: 1 (not at all) - 9 (extremely)"
    elif question.endswith('smoking_mc3'):
        q = "In your opinion, how harmful is smoking to your general health? - Scale: 0 (not at all)-6 (extremely harmful)"
    elif question.endswith('smoking_mc4'):
        q = "How open are you to smoking in the future? - Scale: 1 (not at all)-9 (extremely)"
    elif question.endswith('sexual_practice_mc1'):
        q = "The message makes me more concerned about the health risks of unsafe sexual practices - Scale: 1 (not at all) - 9 (extremely)"
    elif question.endswith('sexual_practice_mc2'):
        q = "The message motivates me to not engage in unsafe sexual behavior - Scale: 1 (not at all) - 9 (extremely)"
    elif question.endswith('sexual_practice_mc3'):
        q = "In your opinion, how harmful is unsafe sexual behavior to your general health? - Scale: 0 (not at all)-6 (extremely harmful)"
    elif question.endswith('sexual_practice_mc4'):
        q = "How open are you to practicing safe sex in the future? - Scale: 1 (not at all)-9 (extremely)"
    else:
        q = question
        print("No specific question found for this row. Returning:")
        print(q)
        return None
    return q
        
def create_prompt(row, topic_responses_dict):
    
    free_form_answer = None
    for q, ans in topic_responses_dict.items():
        if q.endswith('Type in every thought that came to mind viewing this material.'):
            free_form_answer = ans
            break
    filtered_responses = {
        q: a for q, a in topic_responses_dict.items() if q != row['question']
    }
   
    questions = get_mc_questions(row)
    questions = [q for q in questions if q != row['question']]
    other_questions = []
    other_answers = []
    other_answers = [topic_responses_dict[q] for q in topic_responses_dict if not q.endswith('Type in every thought that came to mind viewing this material.')]

    for key, value in topic_responses_dict.items():
        other_questions.append(key)
        other_answers.append(value)
   
    response_lines = [f"Question {i+1}: {q}: {a}" for i, (q, a) in enumerate(zip(questions, other_answers))]
    
    # question_block = "\n".join(response_lines)
    qa_pairs = list(zip(questions, other_answers))
    num_to_include = random.randint(0, len(qa_pairs))
    selected_pairs = random.sample(qa_pairs, num_to_include)
    question_block = "\n".join([f"Question {i+1}: {q}: {a}" for i, (q, a) in enumerate(selected_pairs)])
    
    if row['question']==None or row['response'] == None:
        print("No question/response found")
        return None
    
    prompt = f"You are a helpful assistant trained to interpret user thoughts and feelings and predict how they would react and answer different questions about various health topics.\n"
    # prompt += f"The topic is: {row['topic']}\n\n"
    prompt += f"Five health topics are randomly selected for you from the following list: Nutrition, Vaccination, Mental Health, Substance abuse, COPD, Chronic Diseases, HIV/aids, Sexual Health. "
    # breakpoint()
    if free_form_answer and row['question'].endswith('Type in every thought that came to mind viewing this material.') == False:
        if random.random() < 0.90:
            include_demographics = random.random() < 0.90
            include_personality = random.random() < 0.90

            if not include_demographics and not include_personality:
                if random.random() < 0.5:
                    include_demographics = True
                else:
                    include_personality = True

            if include_demographics:
                prompt += "You are of the following demographics: " + get_demographics(row) + ". "

            if include_personality:
                prompt += "You have the following personality traits: " + get_personality(row) + ". "
                
        if random.random() < 0.75:
            prompt += "You have a " + row['locus'] + ". "
        if random.random() < 0.5:
            prompt += "You first answer baseline questions about each health topic. For the topic of " + row['topic'] + ", you answer as follows: " + get_baseline(row) + "\n"
        if random.random() < 0.3:
            prompt += "You are then shown the following image and you answer the following: " + question_block + "\n"
        if random.random() < 0.5:
            prompt += "You are shown the following image and asked to 'type in every thought that came to mind viewing this material' and you answer as follows: " + free_form_answer + "\n"
            
        prompt += f"Please provide your response to the following question: " + (row['question']) + ": "
        
     
        return prompt
    elif row['question'].endswith('Type in every thought that came to mind viewing this material.'):
        if random.random() < 0.90:
            include_demographics = random.random() < 0.90
            include_personality = random.random() < 0.90

            if not include_demographics and not include_personality:
                if random.random() < 0.5:
                    include_demographics = True
                else:
                    include_personality = True

            if include_demographics:
                prompt += "You are of the following demographics: " + get_demographics(row) + ". "

            if include_personality:
                prompt += "You have the following personality traits: " + get_personality(row) + ". "
                
        if random.random() < 0.75:
            prompt += "You have a " + row['locus'] + ". "
        if random.random() < 0.5:
            prompt += "You first answer baseline questions about each health topic. For the topic of " + row['topic'] + ", you answer as follows: " + get_baseline(row) + "\n"
        if random.random() < 0.3:
            prompt += "You are then shown the following image and you answer the following: " + question_block + "\n"
        
        prompt += f"Given the question: 'type in every thought that came to mind viewing this material.' "
        prompt += f"What would your response be? "
        return prompt
    
    else:
        print("Unknown question type")
        return None
    
  
@lru_cache(maxsize=100)
def load_cached_image(path):
    try:
        return Image.open(path).copy()
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None

def create_dataset(df, only_images=False, validation_split=0.15, num_proc=None):
    # Set default num_proc to 75% of available CPUs
    if num_proc is None:
        num_proc = max(1, int(cpu_count() * 0.75))
    
    # Filter media files with progress
    media_pattern = '.png|.jpg|.jpeg|.gif|.bmp' if only_images else '.png|.jpg|.jpeg|.gif|.bmp|.mp4'
    print("Filtering media files...")
    df = df[df['media_path'].str.contains(media_pattern, na=False, regex=True)]
    
    print("Processing text responses...")
    df.loc[df['current_profession'] == 'Other (please specify):', 'current_profession'] = df['current_profession_text']
    df.loc[df['marital_status'] == 'Other (please specify):', 'marital_status'] = df['marital_status_text']
    df.loc[df['family_status'] == 'Other (please specify):', 'family_status'] = df['family_status_text']

    # Get and split unique media
    print("Splitting dataset...")
    unique_media = df['media_path'].dropna().unique()
    num_val_media = max(1, int(len(unique_media) * validation_split))
    np.random.shuffle(unique_media)
    val_media_set = set(unique_media[:num_val_media])
    
    train_media_set = set(unique_media[num_val_media:])
    
    # Initialize lists
    train_messages, train_media = [], []
    val_messages, val_media = [], []
    
    # Process data with progress bar
    print("Processing samples...")
    groups = list(df.groupby(['ResponseId', 'topic']))
    for (response_id, topic), group in tqdm(groups, desc="Processing groups"):
        topic_responses_dict = {row['question']: row['response'] for _, row in group.iterrows()}

        for _, current_row in group.iterrows():
            prompt = create_prompt(current_row, topic_responses_dict)
            if prompt is None:
                continue
                
            answer = str(current_row['response'])
            media_path = current_row.get('media_path', None)
            user_content = []

            if media_path and isinstance(media_path, str):
                if media_path.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                    # img = load_cached_image(media_path)
                    # if img:
                    user_content.append({"type": "image", "image": media_path})
                elif media_path.lower().endswith(".mp4"):
                    user_content.append({"type": "video_path", "video_path": media_path})

            user_content.append({"type": "text", "text": str(prompt)})
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]}
            ]
            
            if media_path in val_media_set:
                val_messages.append(messages)
                val_media.append(media_path or "")
            else:
                train_messages.append(messages)
                train_media.append(media_path or "")
    
    # Clear cache and shuffle datasets in parallel
    load_cached_image.cache_clear()
    
    print("Creating and shuffling datasets...")
    train_dataset = Dataset.from_dict({
        "conversations": train_messages,
        "images": train_media
    }).shuffle(num_proc=num_proc)
    
    val_dataset = Dataset.from_dict({
        "conversations": val_messages,
        "images": val_media
    }).shuffle(num_proc=num_proc)
    
    print(f"\nDataset prepared with {len(train_dataset)} training and {len(val_dataset)} validation samples.")
    print(f"Training media: {len(train_media_set)}, Validation media: {len(val_media_set)}")
    
    return train_dataset, val_dataset

def create_dataset(df, only_images=False, validation_split=0.15):
    messages_list = []  
    media_objs = []
    train_messages = []
    train_media = []
    val_messages = []
    val_media = []
    
    if only_images:
        # Include both images and videos if only_images is False, otherwise just images
        df = df[df['media_path'].str.contains('.png|.jpg|.jpeg|.gif|.bmp', na=False)]
    else:
        # Include both images and videos
        df = df[df['media_path'].str.contains('.png|.jpg|.jpeg|.gif|.bmp|.mp4', na=False)]
   
    # Replacing "Other (please specify):" with the actual text provided
    df.loc[df['current_profession'] == 'Other (please specify):', 'current_profession'] = df['current_profession_text']
    df.loc[df['marital_status'] == 'Other (please specify):', 'marital_status'] = df['marital_status_text']
    df.loc[df['family_status'] == 'Other (please specify):', 'family_status'] = df['family_status_text']

    unique_media = df['media_path'].dropna().unique()
    num_val_media = max(1, int(len(unique_media) * validation_split))
    
    np.random.shuffle(unique_media)
    val_media_set = set(unique_media[:num_val_media])
    train_media_set = set(unique_media[num_val_media:])
    
    grouped = df.groupby(['ResponseId', 'topic'])
    
    for name, group in grouped:
        response_id, topic = name

        topic_responses_dict = {
            row['question']: row['response'] for _, row in group.iterrows()
        }

        for _, current_row in group.iterrows():
            previous_responses = [
                f"Question: {q}, Answer: {a}"
                for q, a in topic_responses_dict.items()
                if q != current_row['question']
            ]
            previous_responses_text = "; ".join(previous_responses)

            prompt = create_prompt(current_row, topic_responses_dict)
            if prompt is None:
                print("Warning: Prompt generation failed for row.")
                continue
                
            answer = str(current_row['response'])  # Ensure answer is string
            media_path = current_row.get('media_path', None)

            user_content = []
           
            media_obj = None
            
            if media_path and isinstance(media_path, str): 
                try:
                    if media_path.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                        # img = Image.open(media_path)
                        user_content.append({"type": "image", "image": media_path})  
                except (FileNotFoundError, Exception) as e:
                    print(f"Warning: Error loading media at {media_path}: {e}")
                    continue
            else:
                pass

            # Add text prompt
            user_content.append({"type": "text", "text": str(prompt)})

            # Format messages
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]}
            ]
            
            if media_path in val_media_set:
                val_messages.append(messages)
                val_media.append(media_path if media_path else "")
            elif media_path in train_media_set or media_path is None:
                train_messages.append(messages)
                train_media.append(media_path if media_path else "")
    
    print(f"Dataset prepared with {len(train_messages)} training samples and {len(val_messages)} validation samples.")
    print(f"Unique media in training: {len(train_media_set)}, in validation: {len(val_media_set)}")
    
    train_dataset = Dataset.from_dict({
        "conversations": train_messages,
        "images": train_media,  # Changed from "images" to "media" to be more generic
    })
    
    val_dataset = Dataset.from_dict({
        "conversations": val_messages,
        "images": val_media,  # Changed from "images" to "media" to be more generic
    })
    
    return train_dataset, val_dataset


def create_dataset(df, only_images=False, validation_split=0.10, demographic_columns=None, holdout_per_demo=10):
    messages_list = []  
    train_messages = []
    train_media = []
    val_messages = []
    val_media = []
    
    if only_images:
        df = df[df['media_path'].str.contains('.png|.jpg|.jpeg|.gif|.bmp', na=False, regex=True)]
    else:
        df = df[df['media_path'].str.contains('.png|.jpg|.jpeg|.gif|.bmp|.mp4', na=False, regex=True)]
   
    # Get all unique media stimuli
    unique_media = df['media_path'].dropna().unique()
    num_val_media = max(1, int(len(unique_media) * validation_split))
    
    # Split media into train and validation sets
    np.random.shuffle(unique_media)
    val_media_set = set(unique_media[:num_val_media])
    train_media_set = set(unique_media[num_val_media:])
    print(val_media_set)
    # First get unique respondents
    respondent_ids = df['ResponseId'].unique()
    
    if demographic_columns:
        respondents_df = df.drop_duplicates('ResponseId')[['ResponseId'] + demographic_columns]
        
        total_respondents = len(respondents_df)
        target_val_respondents = int(total_respondents * validation_split)
        
        demo_groups = respondents_df.groupby(demographic_columns).ngroups
        adjusted_holdout = max(1, int(target_val_respondents / demo_groups))
        
        val_respondents = set()
        for _, group in respondents_df.groupby(demographic_columns):
            group_size = len(group)
            group_holdout = min(adjusted_holdout, max(1, int(group_size * validation_split)))
            
            if group_size > group_holdout:
                val_sample = group.sample(n=group_holdout, random_state=42)
                val_respondents.update(val_sample['ResponseId'].tolist())
            else:
                val_respondents.update(group['ResponseId'].tolist())
    else:
        # If no demographics specified, do random split
        num_val_respondents = int(len(respondent_ids) * validation_split)
        val_respondents = set(np.random.choice(respondent_ids, size=num_val_respondents, replace=False))
    
    
    
    if demographic_columns:
        demo_tracker = df[['ResponseId'] + demographic_columns].drop_duplicates()
        demo_tracker['in_validation'] = demo_tracker['ResponseId'].isin(val_respondents)
        
        print("\nDemographic Distribution in Validation Set:")
        for demo_col in demographic_columns:
            print(f"\n{demo_col}:")
            val_demo_counts = demo_tracker[demo_tracker['in_validation']][demo_col].value_counts()
            train_demo_counts = demo_tracker[~demo_tracker['in_validation']][demo_col].value_counts()
            
            counts_df = pd.DataFrame({
                'Training': train_demo_counts,
                'Validation': val_demo_counts
            }).fillna(0)
            
            counts_df['% Validation'] = (counts_df['Validation'] / 
                                       (counts_df['Training'] + counts_df['Validation'])) * 100
            print(counts_df.sort_values('Validation', ascending=False))
            
    grouped = df.groupby(['ResponseId', 'topic'])
    
    for name, group in grouped:
        response_id, topic = name
        in_val_respondent = response_id in val_respondents
        
        topic_responses_dict = {
            row['question']: row['response'] for _, row in group.iterrows()
        }

        for _, current_row in group.iterrows():
            prompt = create_prompt(current_row, topic_responses_dict)
            if prompt is None:
                continue
                
            answer = str(current_row['response'])
            media_path = current_row.get('media_path', None)
            in_val_media = media_path in val_media_set if media_path else False

            user_content = []
            
            if media_path and isinstance(media_path, str): 
                try:
                    if media_path.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                        user_content.append({"type": "image", "image": media_path})  
                    elif media_path.lower().endswith(".mp4"):
                        user_content.append({"type": "video_path", "video_path": media_path})
                except (FileNotFoundError, Exception) as e:
                    continue

            user_content.append({"type": "text", "text": str(prompt)})

            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]}
            ]
            
            # Prioritize respondent-based split over media-based split
            if in_val_respondent:
                val_messages.append(messages)
                val_media.append(media_path if media_path else "")
            elif in_val_media:
                # Only add to validation if not already added by respondent split
                # AND if we haven't exceeded our validation size target
                current_val_ratio = len(val_messages) / (len(train_messages) + len(val_messages) + 1)
                if current_val_ratio < validation_split * 1.1:  # Allow 10% overshoot
                    val_messages.append(messages)
                    val_media.append(media_path if media_path else "")
                else:
                    train_messages.append(messages)
                    train_media.append(media_path if media_path else "")
            else:
                train_messages.append(messages)
                train_media.append(media_path if media_path else "")
    
    # Calculate final ratios
    total_samples = len(train_messages) + len(val_messages)
    actual_val_ratio = len(val_messages) / total_samples
    
    print(f"\nDataset prepared with {len(train_messages)} training samples and {len(val_messages)} validation samples.")
    print(f"Validation ratio: {actual_val_ratio:.1%} (target was {validation_split:.1%})")
    print(f"Unique media in training: {len(train_media_set)}, in validation: {len(val_media_set)}")
    
    train_dataset = Dataset.from_dict({
        "conversations": train_messages,
        "images": train_media,
    })
    
    val_dataset = Dataset.from_dict({
        "conversations": val_messages,
        "images": val_media,
    })
    
    return train_dataset, val_dataset


def create_dataset_with_topic_based_holdout(df, demographic_columns, only_images=True):
    if only_images:
        df = df[df['media_path'].str.contains('.png|.jpg|.jpeg|.gif|.bmp', na=False, regex=True)]

    topics = df['topic'].unique()
    heldout_pairs = []

    for topic in topics:
        topic_df = df[df['topic'] == topic]
        
        unique_combos = topic_df.drop_duplicates(['media_path', 'ResponseId'])

        # Picking one (media, respondent) pair per topic
        selected = (
            unique_combos.groupby(demographic_columns)
            .apply(lambda x: x.sample(1, random_state=42))
            .reset_index(drop=True)
        )
        if not selected.empty:
            heldout_pairs.append(selected.iloc[0][['media_path', 'topic']])

    heldout_df = pd.DataFrame(heldout_pairs)
    heldout_set = set([tuple(x) for x in heldout_df[['media_path', 'topic']].values])

    val_df = df[df.apply(lambda row: (row['media_path'], row['topic']) in heldout_set, axis=1)].copy()
    train_df = df[~df.apply(lambda row: (row['media_path'], row['topic']) in heldout_set, axis=1)].copy()

    def build_messages(sub_df):
        messages, media_list = [], []
        for (response_id, topic), group in sub_df.groupby(['ResponseId', 'topic']):
            topic_responses_dict = {row['question']: row['response'] for _, row in group.iterrows()}
            for _, row in group.iterrows():
                prompt = create_prompt(row, topic_responses_dict)
                if not prompt:
                    continue
                answer = str(row['response'])
                media_path = row.get('media_path', None)
                user_content = []
                if media_path:
                    if media_path.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                        user_content.append({"type": "image", "image": media_path})
                    elif media_path.lower().endswith(".mp4"):
                        user_content.append({"type": "video_path", "video_path": media_path})
                user_content.append({"type": "text", "text": prompt})
                messages.append([
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": [{"type": "text", "text": answer}]}
                ])
                media_list.append(media_path)
        return messages, media_list

    train_messages, train_media = build_messages(train_df)
    val_messages, val_media = build_messages(val_df)

    print(f"\nFinal Split:")
    print(f"Held-out media: {val_media}")
    print(f"Training samples: {len(train_messages)}, Unique media: {len(set(train_media))}")
    print(f"Validation samples: {len(val_messages)}, Unique media: {len(set(val_media))}")
    
    for col in demographic_columns:
        print(f"\n{col} distribution:")
        print(val_df[col].value_counts(dropna=False))


    train_dataset = Dataset.from_dict({"conversations": train_messages, "images": train_media})
    val_dataset = Dataset.from_dict({"conversations": val_messages, "images": val_media})

    return train_dataset, val_dataset


def consolidate_categories(df):
    religion_mapping = {
        'Christianity': 'Christian',
        'Islam': 'Muslim',
        'Buddhism': 'Other Religion',
        'Judaism': 'Other Religion',
        'Hinduism': 'Other Religion',
        'Agnostic': 'Other Religion',
        'None of the above': 'Unknown',
        'Prefer not to say': 'Unknown',
        # 'Other (please specify)': 'Other Religion'
    }
    df['religion_cons'] = df['religion'].map(religion_mapping).fillna('Unknown')
    
    race_mapping = {
        'White/Caucasian': 'White',
        'Black/African American': 'Black',
        'Asian': 'Asian',
        'Hispanic/Latino': 'Hispanic',
        'Native American/Alaska Native': 'Indigenous',
        'Native Hawaiian/Other Pacific Islander': 'Indigenous',
        # 'Other (please specify)': 'Other'
    }
    
    df['race_cons'] = df['race_ethnicity'].map(race_mapping)
    
    multiracial_cases = df[df['race_cons'].isna() & df['race_ethnicity'].str.contains(',')]
    df.loc[multiracial_cases.index, 'race_cons'] = (
        multiracial_cases['race_ethnicity'].str.split(',').str[0].map(race_mapping)
    )
    gender_mapping = {
        'Male': 'Male',
        'Female': 'Female',
        'Non-Binary/Third Gender':  'Other',
        'Prefer Not To Say': 'Other',
    }
    
    df['gender_cons'] = df['gender'].map(gender_mapping).fillna('Other')
    
    return df


def main():
    file_path = 'PHORECAST/dataset/final_WE_imgs_full.csv'
    data = load_data(file_path)
    
    data = data[~data['question'].str.endswith('.1', na=False)].copy()
    data = data[~data['question'].str.endswith('V1', na=False)].copy()
    data = data[~data['question'].str.endswith('D1', na=False)].copy()
    data = data[~data['question'].str.endswith('SA1', na=False)].copy()
    data = data[~data['question'].str.endswith('SP1', na=False)].copy()
    data = data[~data['question'].str.endswith('SP_groc1', na=False)].copy()
    data = data[~data['question'].str.endswith('AS1', na=False)].copy()
    
    data.loc[data['current_profession'] == 'Other (please specify):', 'current_profession'] = data['current_profession_text']
    data.loc[data['marital_status'] == 'Other (please specify):', 'marital_status'] = data['marital_status_text']
    data.loc[data['family_status'] == 'Other (please specify):', 'family_status'] = data['family_status_text']
    data.loc[data['religion'] == 'Other (please specify)', 'religion'] = data['religion_text']
    
    demographic_cols = ['gender', 'religion', 'race_ethnicity']
    demographic_cols = ['gender_cons', 'religion_cons', 'race_cons']

    data = consolidate_categories(data)
  
    # train_dataset, val_dataset = create_dataset(data, only_images=True, validation_split=0.10, demographic_columns=demographic_cols)
    train_dataset, val_dataset = create_dataset_with_topic_based_holdout(data, demographic_cols, only_images=True)
  
    train_dataset = train_dataset.shuffle()
    val_dataset = val_dataset.shuffle()
    
    return train_dataset, val_dataset


if __name__ == "__main__":
    train_dataset, val_dataset = main()
    
    print("Training dataset length:", len(train_dataset))
    print("Validation dataset length:", len(val_dataset))
    
    train_save_path = './final_training_dataset' 
    val_save_path = './final_validation_dataset'
    
    train_dataset.save_to_disk(train_save_path)
    val_dataset.save_to_disk(val_save_path)
    # train_dataset.to_csv(train_save_path + '.csv')
    # val_dataset.to_csv(val_save_path + '.csv')
  
  