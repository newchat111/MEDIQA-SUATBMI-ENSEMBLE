def prompt(n, query, response):
    PROMPT = f"""
You are an expert in dermatology and woundcare. Now, your job is to rephrase the patient's question and the response to the question from a doctor.
Your rephrased question and response should not lose any key information. Your rephrased sentence will be used as augmented data so accuracy and completeness is very important.
Therefore, they should not look too similar to the original question or response.
Output your response in strict JSON format: You will need to rephrase the response and the question for {n} times. They must be very different in style but have the same contextual meaning, key information, and completeness.

Here is the query and the response:
1.query: {query}
2.response:{response}


**Important:**
- Output only JSON. Do NOT include explanations or extra text.
- Each object corresponds to one image to rate.
- Do not add extra commas or trailing text.

HERE is YOUR OUTPUT FORMAT
"""
    
    llm_output = []
    for i in range(n):
        i = i + 1
        llm_output_response = {
            f'rephrased_response{i}': f"<your rephrased rephrased_response{i}>",
            f'rephrased_question{i}': f"<your rephrased rephrased_question{i}>"
        }

        llm_output.append(llm_output_response)

    
    PROMPT = PROMPT + "\n" + str(llm_output)
    return PROMPT

if __name__ == "__main__":
    print(prompt(n = 3, query = "aa", response="bb"))





    
