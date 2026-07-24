import os
from openai import OpenAI

class GPT54:
    def __init__(self, model_name_or_path="gpt-5.4", temperature=0.0):
        # 从环境变量读取 OpenAI API Key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("请在 .env 文件中设置 OPENAI_API_KEY")
        base_url = "https://yunwu.ai/v1"

        self.client = OpenAI(api_key=api_key,base_url=base_url)
        self.model = model_name_or_path
        self.temperature = temperature

    def get_completion(self, user_prompt):
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt
                    },
                ],
                temperature=self.temperature,
                timeout=300,
            )
            return completion.choices[0].message.content.strip()

        except Exception as e:
            return f"Error: {str(e)}"