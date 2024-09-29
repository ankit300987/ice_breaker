import os
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

contextInformation = """
Conceptual guide
This section contains introductions to key parts of LangChain.

Architecture
LangChain as a framework consists of a number of packages.

langchain-core
This package contains base abstractions of different components and ways to compose them together. The interfaces for core components like LLMs, vector stores, retrievers and more are defined here. No third party integrations are defined here. The dependencies are kept purposefully very lightweight.

langchain
The main langchain package contains chains, agents, and retrieval strategies that make up an application's cognitive architecture. These are NOT third party integrations. All chains, agents, and retrieval strategies here are NOT specific to any one integration, but rather generic across all integrations.

langchain-community
This package contains third party integrations that are maintained by the LangChain community. Key partner packages are separated out (see below). This contains all integrations for various components (LLMs, vector stores, retrievers). All dependencies in this package are optional to keep the package as lightweight as possible.

Partner packages
While the long tail of integrations is in langchain-community, we split popular integrations into their own packages (e.g. langchain-openai, langchain-anthropic, etc). This was done in order to improve support for these important integrations.

langgraph
langgraph is an extension of langchain aimed at building robust and stateful multi-actor applications with LLMs by modeling steps as edges and nodes in a graph.

LangGraph exposes high level interfaces for creating common types of agents, as well as a low-level API for composing custom flows.

langserve
A package to deploy LangChain chains as REST APIs. Makes it easy to get a production ready API up and running.


First-class streaming support: When you build your chains with LCEL you get the best possible time-to-first-token (time elapsed until the first chunk of output comes out). For some chains this means eg. we stream tokens straight from an LLM to a streaming output parser, and you get back parsed, incremental chunks of output at the same rate as the LLM provider outputs the raw tokens.

Async support: Any chain built with LCEL can be called both with the synchronous API (eg. in your Jupyter notebook while prototyping) as well as with the asynchronous API (eg. in a LangServe server). This enables using the same code for prototypes and in production, with great performance, and the ability to handle many concurrent requests in the same server.

Optimized parallel execution: Whenever your LCEL chains have steps that can be executed in parallel (eg if you fetch documents from multiple retrievers) we automatically do it, both in the sync and the async interfaces, for the smallest possible latency.

Retries and fallbacks: Configure retries and fallbacks for any part of your LCEL chain. This is a great way to make your chains more reliable at scale. We’re currently working on adding streaming support for retries/fallbacks, so you can get the added reliability without any latency cost.

Access intermediate results: For more complex chains it’s often very useful to access the results of intermediate steps even before the final output is produced. This can be used to let end-users know something is happening, or even just to debug your chain. You can stream intermediate results, and it’s available on every LangServe server.

Input and output schemas Input and output schemas give every LCEL chain Pydantic and JSONSchema schemas inferred from the structure of your chain. This can be used for validation of inputs and outputs, and is an integral part of LangServe.

Seamless LangSmith tracing As your chains get more and more complex, it becomes increasingly important to understand what exactly is happening at every step. With LCEL, all steps are automatically logged to LangSmith for maximum observability and debuggability.

LCEL aims to provide consistency around behavior and customization over legacy subclassed chains such as LLMChain and ConversationalRetrievalChain. Many of these legacy chains hide important details like prompts, and as a wider variety of viable models emerge, customization has become more and more important.

If you are currently using one of these legacy chains, please see this guide for guidance on how to migrate.

For guides on how to do specific tasks with LCEL, check out the relevant how-to guides.
"""


if __name__ == '__main__':
    import time
    print('Setting up the template')
    summary_template = """
        Please as an assistant, you should answer user query in the style of an icebreaker. 
        Your answer should be from the context. if the answer is not matching in the context, then please say we could not find answer.
        query: {query}
        context: {context}
    """
    template = PromptTemplate(input_variables=["query", "context"], template=summary_template)
    # llm = setting up model here
    # llm = ChatOllama(model="llama3")
    llm = ChatOllama(model="mistral")
    chain = template | llm | StrOutputParser()


    start_time = time.time()
    print('Setting up the context and executing query please wait')
    result = chain.invoke({"query": "Write a poem on langchain?", "context": contextInformation })
    print('Almost done')
    end_time = time.time()
    print(f"Time taken to run the statement: {end_time - start_time} seconds")
    print(result)