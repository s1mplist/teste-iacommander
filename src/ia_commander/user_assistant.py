from langchain_openai import ChatOpenAI

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool

import requests
import random

class UserAgent:
    def __init__(self,
                llm: ChatOpenAI,
                *args, **kwargs
                 ):
        
        self.llm = llm
        self.chat_history = []
    
    @staticmethod
    def _system_message() -> str:
        return """Você é um agente da OpenAI.
    Se precisar gerar algum número aleatório utilize sempre a tool get_rand_number e use seu resultado em outras tools"""
    
    def _get_prompt_template(self, message:str):
        return ChatPromptTemplate.from_messages(
            messages=[
                ("system", self._system_message()),
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
    
    @staticmethod
    @tool
    def get_comments(id: str = None) -> str:
        """Tool to get comments from our website
        To use this tool, you can pass the id of the comment or simply just call the function to get the list of all comments
        
        You can call the get_rand_number tool if you need to show random comments from the page
        
        Args:
            id: str - The id of comment (OPCIONAL)
        """
        
        url = "https://jsonplaceholder.typicode.com/comments"
        if id:
            url = f"https://jsonplaceholder.typicode.com/comments/{id}"
        
        request = requests.request(method="GET",
                                   url=url)
        return request.text
    
    @staticmethod
    @tool
    def get_rand_number(qtd: int):
        """Tool to get a list of random numbers
        Example: get_rand_number(4) return [1,5,2,6]
        
        Args:
            qtd: int - Amount of random numbers desired"""
        result = random.sample(range(1, 200), qtd)
        return result
    
            
    def run_chat_llm(self, message: str):
        response = self.llm.invoke(message)
    
        print(f"Response: {response}")
        
        result = response.content
        print(f"\nResult: {result}")
        
        return result

    def run_chat_agent(self, message: str):
        toolkit = [self.add, self.get_comments]
        
        prompt = self._get_prompt_template(message=message)
        
        agent = create_tool_calling_agent(llm=self.llm, tools=toolkit, prompt=prompt)
        agent_executor = AgentExecutor(agent=agent, tools=toolkit, verbose=True)
        
        response = agent_executor.invoke(input={
            'chat_history': self.chat_history,
            'input': message
        })
                
        print(f"Response: {response}")
        
        result = response['output']
        print(f"\nResult: {result}")
        
        return result