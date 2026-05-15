import time

_current_logs = []
_log_callback = None

def set_log_callback(callback):
    global _log_callback
    _log_callback = callback

def log_step(step_name, status="start", detail=""):
    global _current_logs, _log_callback
    timestamp = time.strftime("%H:%M:%S")
    log_entry = {"timestamp": timestamp, "step": step_name, "status": status, "detail": detail}
    _current_logs.append(log_entry)
    
    if _log_callback:
        _log_callback(log_entry)
    
    if status == "start":
        print(f"[{timestamp}] ⏳ {step_name}...")
    elif status == "done":
        print(f"[{timestamp}] ✅ {step_name}完成 {detail}")
    elif status == "error":
        print(f"[{timestamp}] ❌ {step_name}失败: {detail}")
    elif status == "info":
        print(f"[{timestamp}] ℹ️  {step_name}: {detail}")

def clear_logs():
    global _current_logs, _log_callback
    _current_logs = []
    _log_callback = None

def get_logs():
    global _current_logs
    return _current_logs
