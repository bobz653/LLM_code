from openai import OpenAI
import os

# Initialize OpenAI client
client = OpenAI(
    # If the environment variable is not configured, replace with your API Key: api_key="sk-xxx"
    # How to get an API Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
    #api_key=os.getenv("DASHSCOPE_API_KEY"),
    #api_key="sk-33fd213b499f4f029f90bdfb72a839ca1",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

reasoning_content = ""
content = ""

is_answering = False

completion = client.chat.completions.create(
    model="qwq-32b",
    messages=[
        {"role": "user", "content": "9.8和9.11哪个大?"}
    ],
    stream=True,
    # Uncomment the following line to return token usage in the last chunk
     stream_options={
         "include_usage": True
     }
)

print("\n" + "=" * 20 + "reasoning content" + "=" * 20 + "\n")
chunk_num = 0
for chunk in completion:
    # If chunk.choices is empty, print usage
    chunk_num += 1
    print("chunck num: {}\n".format(chunk_num))
    if not chunk.choices:
        print("\nUsage:")
        print(chunk.usage)
    else:
        delta = chunk.choices[0].delta
        # Print reasoning content
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
            print(delta.reasoning_content, end='', flush=True)
            reasoning_content += delta.reasoning_content
        else:
            if delta.content != "" and is_answering is False:
                print("\n" + "=" * 20 + "content" + "=" * 20 + "\n")
                is_answering = True
            # Print content
            print(delta.content, end='', flush=True)
            content += delta.content
