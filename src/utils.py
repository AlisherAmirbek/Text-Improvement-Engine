from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from scipy.spatial.distance import cosine
import csv
from sentence_transformers import SentenceTransformer, util

if torch.cuda.is_available():
    print("GPU is available")
    device = torch.device("cuda")
else:
    print("GPU is not available, using CPU instead")
    device = torch.device("cpu")

decoder = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

encoder = SentenceTransformer('all-MiniLM-L6-v2')

def load_standard_phrases(file_path):
    standard_phrases = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            standard_phrases.extend(row)
    return standard_phrases
    
def get_embedding(text):
    embedding = encoder.encode(text, convert_to_tensor=True)
    return embedding

def find_most_similar(input_text, standard_phrases):
    input_embedding = get_embedding(input_text)
    max_similarity = -1 
    most_similar_phrase = ""
    
    for phrase in standard_phrases:
        phrase_embedding = get_embedding(phrase)
        cos_sim = util.cos_sim(input_embedding, phrase_embedding)
        
        similarity_score = cos_sim.item()
        
        if similarity_score > max_similarity:
            max_similarity = similarity_score
            most_similar_phrase = phrase
    
    return most_similar_phrase, max_similarity

def generate_revised_sentence(original_sentence, suggested_improvement):
    prompt = f"Instruct: Revise this sentence to incorporate the suggested improvement while maintaining its original meaning:\nOriginal Sentence: {original_sentence}\nRevise sentence using this phrase: {suggested_improvement}\nOutput:"
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=64, return_attention_mask=False).to(device)
    outputs = decoder.generate(**inputs, max_length=128)
    revised_sentence = tokenizer.batch_decode(outputs)[0]
    
    revised_sentence = revised_sentence.split("Output:")[1].strip()
    
    return revised_sentence
