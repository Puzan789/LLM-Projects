# import json
# from settings import get_llm_response
# from dataprepprompt import create_prompt
# from mytextsplitters import load_text,Split_Documents


# # Main process
# def main(file_path):
#     qna_dataset = []
#     documents = load_text(file_path)
#     # Split the text into smaller chunks
#     split_text = Split_Documents(documents)

#     # Process each chunk to generate Q&A pairs
#     for chunk in split_text:
#         # Generate prompt for Q&A pairs
#         try:
#             prompt = create_prompt(chunk)
#             # Get LLM response using the prompt
#             response_json= get_llm_response(prompt)
#             return json.loads(response_json)

#             # Parse the LLM response to extract Q&A pairs
#         except json.JSONDecodeError :
#             try:
#                  # Clean up the response if it was wrapped in extra formatting like ```json\n or ```
#                     cleaned_text = response_json.strip("```json\n").strip("```")
#                     print("Cleaned text:", cleaned_text)  # Debugging

#                     # Return the cleaned and loaded JSON content
#                     return json.loads(cleaned_text)
#             except json.JSONDecodeError as json_err:
#                     print(f"JSON decoding error after cleaning: {json_err}")
#                     return []
#         except Exception as e:
#              print(f"Error processing chunk: {e}")
#              continue
#     # Save the dataset to a JSON file
#     with open("qa_dataset.json", "w") as f:
#         json.dump(qna_dataset, f, indent=2)

# if __name__ == "__main__":
#     # Process the file
#     file_path = "text/constitution.txt"
#     main(file_path)


import json
from settings import get_llm_response
from dataprepprompt import create_prompt
from mytextsplitters import Load_text, Split_Documents

# Helper function to append JSON in a valid format
def append_to_json_file(file_path, data):
    try:
        with open(file_path, "r+") as f:
            try:
                # Try to load existing data
                existing_data = json.load(f)
            except json.JSONDecodeError:
                # If the file is empty or has invalid JSON, start fresh with an empty list
                existing_data = []

            # Move the cursor to the beginning of the file to overwrite with updated data
            f.seek(0)

            # Append the new data to the existing data
            existing_data.append(data)

            # Write the updated data back to the file
            json.dump(existing_data, f, indent=2)

            # Truncate the file in case the new data is smaller than the old one
            f.truncate()

    except FileNotFoundError:
        # If the file doesn't exist, create it and write the first entry
        with open(file_path, "w") as f:
            json.dump([data], f, indent=2)

# Main process
def main(file_path, output_file="qa_dataset.json"):
    documents = Load_text(file_path)
    
    # Split the text into smaller chunks
    split_text = Split_Documents(documents)

    # Process each chunk to generate Q&A pairs
    for chunk in split_text:
        try:
            prompt = create_prompt(chunk)
            # Get LLM response using the prompt
            response_json = get_llm_response(prompt)

            try:
                # Parse the LLM response to extract Q&A pairs
                qna_pairs = json.loads(response_json)
            except json.JSONDecodeError:
                # Clean up the response if it was wrapped in extra formatting like ```json\n or ```
                cleaned_text = response_json.strip("```json\n").strip("```")
                print("Cleaned text:", cleaned_text)  # Debugging

                # Attempt to load cleaned JSON
                qna_pairs = json.loads(cleaned_text)

            # Append the Q&A pairs to the JSON file after each chunk
            append_to_json_file(output_file, qna_pairs)

        except json.JSONDecodeError as json_err:
            print(f"JSON decoding error after cleaning: {json_err}")
            continue  # Skip this chunk if there's an issue
        except Exception as e:
            print(f"Error processing chunk: {e}")
            continue

if __name__ == "__main__":
    # Process the file
    file_path = "text/constitution.txt"
    main(file_path)
