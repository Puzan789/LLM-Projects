def create_prompt(chunk):
    """
    Generates a prompt to create Q&A pairs from a text passage.
    """
    prompt = f"""
    You are an expert in the Nepali constitution tasked with creating question-answer pairs from the following passage. 
    Focus on key facts, important points, and relevant acts mentioned. Ensure the questions clarify the acts, concepts, and their implications.

    Format the output as a JSON array of objects, each with 'question' and 'answer' keys. Ensure the JSON is valid.

    Passage: "{chunk}"

    Output format (in JSON):

    [
        {{
            "question": "Your question here",
            "answer": "The corresponding concise answer."
        }},
        ...
    ]
    """
    return prompt