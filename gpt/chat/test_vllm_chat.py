#!/usr/bin/env python3
"""
Minimal chat interface for testing vLLM server endpoint
"""

import requests
import json
import os
from datetime import datetime
from pathlib import Path

# vLLM server endpoint (default)
BASE_URL = "http://localhost:8000"

# Create logs directory
LOG_DIR = Path("vllm_logs")
LOG_DIR.mkdir(exist_ok=True)

def get_models():
    """Get available models from the server"""
    try:
        response = requests.get(f"{BASE_URL}/v1/models")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching models: {e}")
        return None

def save_log(log_type, data):
    """Save request/response to timestamped log file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = LOG_DIR / f"{timestamp}_{log_type}.json"
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    return filename

def tokenize_text(text, model_name):
    """Tokenize text using vLLM's tokenize endpoint"""
    url = f"{BASE_URL}/tokenize"
    
    payload = {
        "prompt": text,
        "model": model_name
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"\n[Tokenization error: {e}]")
        return None

def chat_completion(model_name, messages):
    """Send chat completion request to vLLM server"""
    url = f"{BASE_URL}/v1/chat/completions"
    
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 8192,  # High limit to capture full reasoning + response
        "stream": False,
    }
    
    # Calculate approximate token count for debugging
    approx_tokens = sum(len(msg["content"].split()) * 1.3 for msg in messages)
    
    # Save request with extra debug info
    request_log = {
        "url": url,
        "payload": payload,
        "timestamp": datetime.now().isoformat(),
        "debug_info": {
            "num_messages": len(messages),
            "last_message_length": len(messages[-1]["content"]) if messages else 0,
            "approx_total_tokens": int(approx_tokens),
            "last_message_preview": messages[-1]["content"][:100] if messages else ""
        }
    }
    req_file = save_log("request", request_log)
    print(f"[Request saved: {req_file}]")
    print(f"[Debug: ~{int(approx_tokens)} tokens, msg_len={len(messages[-1]['content'])} chars]")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        
        # Check if response is None or empty before parsing
        if response.text == "None" or not response.text:
            print("⚠️  Server returned 'None' or empty response")
            error_log = {
                "error": "Server returned None or empty response",
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "raw_text": response.text,
                "timestamp": datetime.now().isoformat()
            }
            err_file = save_log("error", error_log)
            print(f"[Error saved: {err_file}]")
            return None
            
        response.raise_for_status()
        response_data = response.json()
        
        # Save successful response
        response_log = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": response_data,
            "timestamp": datetime.now().isoformat()
        }
        resp_file = save_log("response", response_log)
        print(f"[Response saved: {resp_file}]")
        
        return response_data
    except requests.exceptions.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        print(f"Raw response text: {response.text[:500]}")
        
        error_log = {
            "error": f"JSON decode error: {str(e)}",
            "response_status": response.status_code,
            "response_headers": dict(response.headers),
            "response_text": response.text,
            "timestamp": datetime.now().isoformat()
        }
        err_file = save_log("error", error_log)
        print(f"[Error saved: {err_file}]")
        return None
        
    except requests.exceptions.RequestException as e:
        print(f"Error during chat completion: {e}")
        
        # Save error response
        error_log = {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response content: {e.response.text[:500]}")
            error_log["response_status"] = e.response.status_code
            error_log["response_headers"] = dict(e.response.headers)
            error_log["response_body"] = e.response.text
        
        err_file = save_log("error", error_log)
        print(f"[Error saved: {err_file}]")
        
        return None

def main():
    print("🔍 Checking vLLM server connection...")
    print(f"Server URL: {BASE_URL}")
    print("-" * 50)
    
    # Get available models
    models_response = get_models()
    if not models_response:
        print("❌ Could not connect to vLLM server. Make sure it's running on localhost:8000")
        return
    
    models = models_response.get("data", [])
    if not models:
        print("❌ No models available on the server")
        return
    
    # Use the first available model
    model_name = models[0]["id"]
    print(f"✅ Connected! Using model: {model_name}")
    print("-" * 50)
    
    # Initialize conversation history
    messages = []
    
    print("\n💬 Chat Interface (type 'quit' to exit)")
    print(f"📁 Logs will be saved to: {LOG_DIR.absolute()}")
    print("=" * 50)
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        # Add user message to history
        messages.append({"role": "user", "content": user_input})
        
        # Tokenize prompt for debugging
        prompt_tokenize = tokenize_text(f"User: {user_input}", model_name)
        if prompt_tokenize:
            prompt_tokens = prompt_tokenize.get("tokens", [])
            print(f"\n🔤 [Tokenized prompt - {len(prompt_tokens)} tokens:]")
            print(f"Prompt token IDs: {prompt_tokens}")
        
        # Get response from vLLM
        print("\nModel: ", end="", flush=True)
        response = chat_completion(model_name, messages)
        
        if response:
            # Extract and print the response
            message = response["choices"][0]["message"]
            content = message["content"]
            
            # Check for reasoning content (CoT)
            if "reasoning_content" in message and message["reasoning_content"]:
                print("\n📝 [Model's reasoning/CoT:]")
                print("-" * 40)
                print(message["reasoning_content"][:1000])  # Show first 1000 chars
                if len(message["reasoning_content"]) > 1000:
                    print(f"... [truncated, full reasoning: {len(message['reasoning_content'])} chars]")
                print("-" * 40)
                print("\n💬 [Model's response:]")
            
            print(content)
            
            # Tokenize full conversation (what we'll save)
            full_text = f"User: {user_input}\n\nAssistant: {content}"
            full_tokenize = tokenize_text(full_text, model_name)
            if full_tokenize:
                full_tokens = full_tokenize.get("tokens", [])
                print(f"\n🔢 [Full tokenized conversation - {len(full_tokens)} tokens:]")
                print(f"All token IDs: {full_tokens}")
            
            # Add assistant message to history
            messages.append({"role": "assistant", "content": content})
            
            # Print token usage and finish reason
            if "usage" in response:
                usage = response["usage"]
                finish_reason = response["choices"][0].get("finish_reason", "unknown")
                print(f"\n[Tokens - Prompt: {usage['prompt_tokens']}, "
                      f"Completion: {usage['completion_tokens']}, "
                      f"Total: {usage['total_tokens']}, "
                      f"Finish: {finish_reason}]")
        else:
            print("❌ Failed to get response from server")
            # Remove the last user message since we didn't get a response
            messages.pop()

if __name__ == "__main__":
    main()