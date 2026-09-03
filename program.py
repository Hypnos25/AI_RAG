from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser 

from dotenv import load_dotenv

load_dotenv()
# โหลดเอกสาร
loader = TextLoader("data.txt",encoding="utf-8")
documents = loader.load()
# print(documents)

# แบ่งส่วนย่อย
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 100,chunk_overlap = 50)
chunks = text_splitter.split_documents(documents)

# for i , chunk in enumerate(chunks):
#     print(f" chunk : {i+1 , {chunk.page_content}}")

#  แปลงข้อมูลเป็นเวกเตอร์
embedding = OpenAIEmbeddings()
# เก็บข้อมูลลง vector store
vectorstore = FAISS.from_documents(chunks,embedding)
#  ตัวดึ store ไปใช้่งาน
retrievers = vectorstore.as_retriever()

# prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system","ใช้ข้อมูลในเอกสารในการตอบคำถามให้สั้นกระชับด้วยความสุภาพเป็นกันเอง"),
    ("human","คำถาม : {question} , ข้อมูลที่เกี่ยวของ : {context}")
])

#  model
llm  =ChatOpenAI(model="gpt-4o-mini")

# chain
rag_chain = (
    {"context" : retrievers,"question" :RunnablePassthrough() }
    |prompt
    |llm
    |StrOutputParser()
)
respons =rag_chain.invoke("กะแดกงา มีวัตถุเเละวิธีการทำยังไง")
print(respons)