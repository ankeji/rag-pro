'''
Author: ankeji ankeji1995@163.com
Date: 2026-05-12 15:09:09
LastEditors: ankeji ankeji1995@163.com
LastEditTime: 2026-05-13 14:01:48
FilePath: \AI_Projects\my_agent\rag-pro\mainAgentGraph.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# ================= LangChain v1 生产级 Tool Calling（无Memory，官方推荐） =================
import os
from dotenv import load_dotenv
load_dotenv()

# ================= LangSmith 配置 =================
os.environ["LANGCHAIN_PROJECT"] = "rag-pro-agent-graph"  # 项目名称

from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END  # 新增LangGraph核心类
from session_store import get_session_history, save_session_history
from typing_extensions import TypedDict

class WorkflowState(TypedDict):
    messages: list  # 会话历史
    session_id: str  # 会话ID
    end: bool = False  # 工作流结束标识

# 你的工具
from tools import get_collection_count, clear_collection, import_pdf_files, query_knowledge_base, check_tool_permission
from config import BASE_URL, KEY

# 工具列表
tools = [get_collection_count, clear_collection, import_pdf_files, query_knowledge_base]

# LLM
llm = ChatOpenAI(model="glm-5.1", api_key=KEY, base_url=BASE_URL.replace("/chat/completions", ""))

# ====================== 生产级 Prompt（官方标准，无Memory） ======================
prompt = ChatPromptTemplate.from_messages([
        ("system", """你是企业知识库助手，严格遵循以下规则：
1.  功能范围：仅可查询表数量、清空表、导入PDF文件、查询知识库内容，不执行超出范围的操作；
2.  权限校验：调用clear_collection、import_pdf_files前，必须先通过check_tool_permission函数校验权限，无权限直接返回提示，不执行后续操作；
3.  工具调用：优先判断用户需求对应哪个工具，不重复调用、不调用无关工具；
4.  流程规范：导入PDF后，需自动调用get_collection_count校验导入效果；清空表后，需自动调用get_collection_count确认清空结果；
5.  错误处理：工具调用失败时，返回明确错误提示，不重复重试，不泄露代码/配置信息。"""),
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
    max_iterations=5,  # 生产级优化：限制最大调用次数，防止无限循环
    early_stopping_method="force",  # 无法判断时强制终止，返回友好提示
    # 完全删除memory，严格遵循官方无Memory架构，会话历史由Redis手动管控
)


# ================= LangGraph 工作流编排（核心优化：管控工具调用流程，避免混乱） =================
# 定义工作流节点（每个节点对应一个核心操作，复用现有逻辑，不新增工具）
def permission_check_node(state):
    """权限校验节点：调用工具前先校验权限，适配高危工具管控"""
    messages = state["messages"]
    session_id = state["session_id"]
    user_input = messages[-1].content  # 获取当前用户输入
    
    # 判断是否需要调用高危工具，触发权限校验
    high_risk_tools = ["clear_collection", "import_pdf_files"]
    # 提取Agent计划调用的工具（简化逻辑，可根据实际需求优化）
    tool_to_call = None
    if "清空" in user_input:
        tool_to_call = "clear_collection"
    elif "导入" in user_input:
        tool_to_call = "import_pdf_files"
    
    # 无需权限校验的场景（安全工具），直接进入Agent执行节点
    if not tool_to_call:
        return {"messages": messages, "session_id": session_id, "skip_permission": True}
    
    # 执行权限校验（调用check_tool_permission，传入session_id关联用户）
    # 这里简化获取用户角色：实际场景可通过session_id从Redis/数据库查询真实角色
    user_role = "admin" if "admin" in session_id else "user"
    print(user_role, tool_to_call, session_id, '权限校验')
    has_permission = check_tool_permission(tool_name=tool_to_call, user_role=user_role)
    
    if not has_permission:
        # 无权限，直接返回提示，终止工作流
        messages.append(AIMessage(content=f"权限不足！您无权限执行【{tool_to_call}】操作，请联系管理员。"))
        return {"messages": messages, "session_id": session_id, "end": True}
    
    # 有权限，进入Agent执行节点
    return {"messages": messages, "session_id": session_id, "skip_permission": False}

def agent_execute_node(state):
    """Agent执行节点：调用Agent处理用户需求，执行工具调用"""
    messages = state["messages"]
    session_id = state["session_id"]
    
    # 执行Agent（和原有调用逻辑一致，无改动）
    res = agent_executor.invoke({"messages": messages})
    
    # 追加Agent回答到会话历史
    messages.append(AIMessage(content=res["output"]))
    return {"messages": messages, "session_id": session_id}

def session_save_node(state):
    """会话保存节点：将最新会话历史存入Redis，确保持久化"""
    messages = state["messages"]
    session_id = state["session_id"]
    
    # 调用原有Redis保存逻辑，无改动
    save_session_history(session_id, messages)
    return {"messages": messages, "session_id": session_id}


# 构建LangGraph工作流（串联节点，定义执行流程）
graph = StateGraph(WorkflowState)

# 1. 添加工作流节点（唯一标识，对应上述节点函数）
graph.add_node("permission_check", permission_check_node)  # 第一步：权限校验
graph.add_node("agent_execute", agent_execute_node)        # 第二步：Agent执行工具调用
graph.add_node("session_save", session_save_node)          # 第三步：会话保存到Redis

# 2. 定义节点执行顺序（边逻辑，管控流程）
# 入口节点：先执行权限校验
graph.add_edge(START, "permission_check")

# 权限校验后，无论是否跳过权限，都进入Agent执行节点
graph.add_conditional_edges(
    "permission_check",
    # 分支判断：若end为True（无权限），直接结束；否则进入Agent执行
    lambda state: END if state.get("end") else "agent_execute"
)

# Agent执行后，进入会话保存节点，保存完成后结束工作流
graph.add_edge("agent_execute", "session_save")
graph.add_edge("session_save", END)

# 3. 编译工作流（生成可执行的工作流实例）
workflow = graph.compile()

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
SESSION_ID = "user"

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
        res = workflow.invoke({
            "messages": messages,
            "session_id": SESSION_ID
        })

        # 4. 提取最终会话消息，用于终端输出
        final_messages = res["messages"]
        last_answer = final_messages[-1].content

        print("\n" + "=" * 50)
        print(f"✅ 回答：{last_answer}")
        print("=" * 50)
    
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，退出程序")
        break
    except Exception as e:
        print(f"\n❌ 发生错误：{type(e).__name__}: {str(e)}")
        continue