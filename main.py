from langchain import HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="tiiuae/falcon-7b-instruct",
    task="text-generation",
    model_kwargs={"temperature": 0, "max_length": 64, "trust_remote_code": True},
)
