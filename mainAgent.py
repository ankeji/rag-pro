# ================= LangChain Tool Calling 集成 =================
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "rag-pro-agent"  # 项目名称

from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from tools import get_collection_count, clear_collection, import_pdf_files, query_knowledge_base
from config import BASE_URL, KEY

tools = [get_collection_count, clear_collection, import_pdf_files, query_knowledge_base]

# LLM
llm = ChatOpenAI(model="glm-5.1", api_key=KEY, base_url=BASE_URL.replace("/chat/completions", ""))

# 智能体
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是企业知识库助手，可以查询表数量、清空表、导入PDF文件、查询知识库内容。"),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)

# ================= 终端问答循环 =================
print("🤖 企业知识库助手已启动！")
print("可用功能：")
print("  1. 查询表数据量（如：handbook表有多少数据？）")
print("  2. 清空表（如：清空handbook表）")
print("  3. 导入PDF文件（如：把docs目录下的所有pdf文件导入到ChromaDB中？）")
print("  4. 查询知识库（如：新员工入职需要哪些培训？）")
print("  输入 'quit' 或 'exit' 退出")
print("=" * 50)

while True:
    try:
        user_input = input("\n💬 请输入问题：")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 再见！")
            break
        
        if not user_input.strip():
            continue
            
        print("⏳ 思考中...")
        
        # 执行 Agent
        res = agent_executor.invoke({"input": user_input})
        
        print("\n" + "=" * 50)
        print(f"✅ 回答：{res.get('output', '无输出')}")
        print("=" * 50)
        
        # 可选：显示更多信息
        show_details = input("\n🔍 显示详细信息？(y/N): ").lower()
        if show_details == 'y':
            print("\n📋 完整结果：")
            for key, value in res.items():
                if key != 'output':
                    print(f"  {key}: {value}")
    
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，退出程序")
        break
    except Exception as e:
        print(f"\n❌ 发生错误：{type(e).__name__}: {str(e)}")
        continue