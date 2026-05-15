# ================= LangChain v1 生产级 Tool Calling（无Memory，官方推荐） =================
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "rag-pro-agent-chain"  # 项目名称

from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from session_store import get_session_history, save_session_history

# 你的工具
from tools import get_collection_count, clear_collection, import_pdf_files, query_knowledge_base
from config import BASE_URL, KEY

# 工具列表
tools = [get_collection_count, clear_collection, import_pdf_files, query_knowledge_base]

# LLM
llm = ChatOpenAI(model="glm-5.1", api_key=KEY, base_url=BASE_URL.replace("/chat/completions", ""))

# ====================== 生产级 Prompt（官方标准，无Memory） ======================
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是企业知识库助手，可以查询表数量、清空表、导入PDF文件、查询知识库内容。"),
    MessagesPlaceholder(variable_name="messages"),  # 手动传入对话历史
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 构建 Agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    handle_parsing_errors=True,
    # 注意：这里 完全删除 memory！！！
)

# ================= 终端问答循环 =================
print("🤖 企业知识库助手已启动！【生产级v1架构】")
print("可用功能：")
print("  1. 查询表数据量（如：handbook表有多少数据？）")
print("  2. 清空表（如：清空handbook表）")
print("  3. 导入PDF文件（如：把docs目录下的所有pdf文件导入到ChromaDB中？）")
print("  4. 查询知识库（如：新员工入职需要哪些培训？）")
print("  输入 'quit' 或 'exit' 退出")
print("=" * 50)

# 固定用一个用户会话（前端可传 userId 作为 session_id）
SESSION_ID = "user_default456789"

while True:
    try:
        user_input = input("\n💬 请输入问题：")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 再见！")
            break
        
        if not user_input.strip():
            continue
            
        print("⏳ 思考中...")

        # 1. 获取历史消息
        messages = get_session_history(SESSION_ID)
        
        # 2. 加入用户当前问题
        messages.append(HumanMessage(content=user_input))

        # 3. 执行 Agent（生产级标准调用）
        res = agent_executor.invoke({
            "messages": messages
        })

        # 4. 把回答加入历史
        messages.append(AIMessage(content=res["output"]))

        # 关键：存入 Redis
        save_session_history(SESSION_ID, messages)

        print("\n" + "=" * 50)
        print(f"✅ 回答：{res.get('output', '无输出')}")
        print("=" * 50)
    
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，退出程序")
        break
    except Exception as e:
        print(f"\n❌ 发生错误：{type(e).__name__}: {str(e)}")
        continue