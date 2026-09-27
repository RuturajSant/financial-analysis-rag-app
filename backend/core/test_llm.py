from backend.core.llm_provider import extract_text, get_chat_llm
test_prompt = "This is a test prompt. count 1 to 5 backwards"

llm=get_chat_llm(temperature=0)

response = llm.invoke(test_prompt)
print(response)
answer = extract_text(response.content).strip()
print(answer)