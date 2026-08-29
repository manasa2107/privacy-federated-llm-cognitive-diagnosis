import os
import time
from groq import Groq
from openai import OpenAI
from anthropic import Anthropic

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

def diagnose_student(student_responses, concepts, entity_id=0):
    n_concepts = len(concepts)
    base_prompt = f"""Given student responses: {student_responses}
And knowledge concepts: {concepts}

Estimate the student's knowledge state as a probability vector [0-1] for EXACTLY {n_concepts} concepts.
Return ONLY a Python list with EXACTLY {n_concepts} numbers, nothing else.

Example format for {n_concepts} concepts: {[0.5] * n_concepts}
"""
    
    try:
        if entity_id == 0:
            prompt = f"You are an educational assessor.\n{base_prompt}"
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.5
            )
            response = chat_completion.choices[0].message.content
            
        elif entity_id == 1:
            prompt = f"You are an educational assessor with strong mathematical reasoning.\n{base_prompt}"
            
            # Retry logic for rate limits
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    chat_completion = openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.5
                    )
                    response = chat_completion.choices[0].message.content
                    break
                except Exception as e:
                    if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                        import time
                        wait_time = (2 ** attempt)
                        print(f"Rate limit hit, waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise
            
        else:
            prompt = f"You are a precise educational assessor.\n{base_prompt}"
            message = anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                temperature=0.5,
                messages=[{"role": "user", "content": prompt}]
            )
            response = message.content[0].text
        
        import re
        response = response.replace("```python", "").replace("```", "").strip()
        match = re.search(r'\[[\d\.,\s]+\]', response)
        if match:
            knowledge_state = eval(match.group())
        else:
            knowledge_state = eval(response.strip())
        
        # CRITICAL FIX: Ensure correct length
        if len(knowledge_state) != n_concepts:
            print(f"Warning: Entity {entity_id} returned {len(knowledge_state)} values instead of {n_concepts}, fixing...")
            if len(knowledge_state) < n_concepts:
                knowledge_state.extend([0.5] * (n_concepts - len(knowledge_state)))
            else:
                knowledge_state = knowledge_state[:n_concepts]
        
        return knowledge_state
        
    except Exception as e:
        print(f"Error in Entity {entity_id}: {e}")
        return [0.5] * n_concepts
def get_llm_prediction(student_responses, concepts, model_name):
    """
    Wrapper function for compatibility with new code
    Maps model names to entity IDs
    """
    model_map = {
        'llama': 0,
        'gpt4': 1,
        'claude': 2
    }
    entity_id = model_map.get(model_name, 0)
    return diagnose_student(student_responses, concepts, entity_id)
