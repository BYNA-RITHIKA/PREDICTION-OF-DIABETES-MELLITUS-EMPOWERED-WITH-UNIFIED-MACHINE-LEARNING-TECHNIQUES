
def chatbot(argument):
    openai.api_key = "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"
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=argument,
    temperature=0.5,
    max_tokens=1024,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    answer=str(response.choices[0].text)
    return answer
print(chatbot("what is normal glucose level?"))