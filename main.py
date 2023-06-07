from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
import time

start = time.time()

llm = HuggingFacePipeline.from_model_id(
    model_id="tiiuae/falcon-7b-instruct",
    task="text-generation",
    model_kwargs={"temperature": 0, "max_length": 64, "trust_remote_code": True},
)

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What is electroencephalography?"

print(llm_chain.run(question))

end = time.time()

print(end - start)
