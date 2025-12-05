import requests, json

a = "sk-or-v1-7995770d42"
b = "78c5cbb2d0ab4335adf3"
c = "b69dfff451870ce8"
d = "9f73cf8b7a9544ce02"

OPENROUTER_API_KEY = a+b+c+d
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"





class LLM_API:
    def __init__(self):
        self.model = "x-ai/grok-4-fast"
        self.headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}","Content-Type": "application/json"}
        self.payload = {"model":self.model, "messages":[], "stream":True}
    
    def set_system_prompt(self, system_prompt: str):
        self.payload['messages'].append({"role":"system", "content":system_prompt})
    
    def add_message(self, role: str, content: str):
        self.payload['messages'].append({"role":role, "content":content})
    
    def clear_messages(self):
        self.payload['messages'] = []
    
    def get_response(self):
        response_text = ""
        response = requests.post(f"{OPENROUTER_BASE_URL}/chat/completions", headers=self.headers, json=self.payload, stream=True)
        # print(self.payload)
        response.raise_for_status()
        for chunk in response.iter_lines():
            if chunk:
                data = chunk.decode('utf-8')    
                if data.startswith('data: '):
                    data = data[6:]
                    if data.strip() != '[DONE]':
                        data = json.loads(data)
                        chunk = (data['choices'][0]['delta']['content'])
                        if chunk:
                            print (chunk, end='', flush=True)
                            response_text += chunk
        return response_text
    def start_chat(self):
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["q", "quit", "exit"]:
                break
            self.add_message("user", user_input)
            response = self.get_response()
            # print("\nAssistant: " + response)
            self.clear_messages()


llm_api = LLM_API()
llm_api.set_system_prompt("You are not a helpfull assistant. you always provide joke answer. you always provide wrong answer.")
llm_api.start_chat()