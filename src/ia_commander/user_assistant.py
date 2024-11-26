from langchain_openai import ChatOpenAI

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool

class UserAgent:
    def __init__(self,
                llm: ChatOpenAI,
                *args, **kwargs
                 ):
        
        self.llm = llm
        self.chat_history = []
    
    def _get_prompt_template(self, message:str):
        return ChatPromptTemplate.from_messages(
            messages=[
                ("system", "Você é um agente da OpenAI"),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}")
                ]
        )       
        
    @staticmethod     
    @tool(return_direct=True)
    def add(a: int, b:int) -> int:
        "Tool for do a add (plus) operation in Math"
        return a + b
            
    def run_chat_llm(self, message: str):
        response = self.llm.invoke(message)
    
        print(f"Response: {response}")
        
        result = response.content
        print(f"\nResult: {result}")
        
        return result

    def run_chat_agent(self, message: str):
        toolkit = [self.add]
        
        prompt = self._get_prompt_template(message=message)
        
        agent = create_tool_calling_agent(llm=self.llm, tools=toolkit, prompt=prompt)
        agent_executor = AgentExecutor(agent=agent, tools=toolkit, verbose=False)
        
        response = agent_executor.invoke(input={
            'chat_history': self.chat_history,
            'input': message
        })
                
        print(f"Response: {response}")
        
        result = response['output']
        print(f"\nResult: {result}")
        
        return result