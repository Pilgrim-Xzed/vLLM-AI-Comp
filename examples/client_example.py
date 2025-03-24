#!/usr/bin/env python3
import argparse
import json
import os
import requests
import sseclient
import time
from typing import List, Dict, Optional, Union

class MistralClient:
    """
    A client for interacting with the Mistral Inference Server API.
    Compatible with the OpenAI API format.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
    
    def list_models(self) -> Dict:
        """List available models."""
        response = self.session.get(f"{self.base_url}/v1/models")
        response.raise_for_status()
        return response.json()
    
    def chat_completion(
        self,
        messages: List[Dict],
        model: str = "mistralai/mistral-7b",
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = None,
    ):
        """
        Generate a chat completion using the chat/completions endpoint.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: Model to use
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            stop: Custom stop sequences
            
        Returns:
            Response from the API or generator if streaming is enabled
        """
        url = f"{self.base_url}/v1/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
            
        if stop is not None:
            payload["stop"] = stop
        
        if stream:
            response = self.session.post(url, json=payload, stream=True)
            response.raise_for_status()
            client = sseclient.SSEClient(response)
            
            for event in client.events():
                if event.data == "[DONE]":
                    break
                    
                try:
                    yield json.loads(event.data)
                except json.JSONDecodeError:
                    pass
        else:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            return response.json()
    
    def completion(
        self,
        prompt: Union[str, List[str]],
        model: str = "mistralai/mistral-7b",
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = None,
    ):
        """
        Generate a text completion using the completions endpoint.
        
        Args:
            prompt: Text prompt or list of prompts
            model: Model to use
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            stop: Custom stop sequences
            
        Returns:
            Response from the API or generator if streaming is enabled
        """
        url = f"{self.base_url}/v1/completions"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
            
        if stop is not None:
            payload["stop"] = stop
        
        if stream:
            response = self.session.post(url, json=payload, stream=True)
            response.raise_for_status()
            client = sseclient.SSEClient(response)
            
            for event in client.events():
                if event.data == "[DONE]":
                    break
                    
                try:
                    yield json.loads(event.data)
                except json.JSONDecodeError:
                    pass
        else:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            return response.json()


def main():
    """Simple CLI interface for the Mistral client."""
    parser = argparse.ArgumentParser(description="Mistral Inference Client")
    
    parser.add_argument(
        "--base_url", 
        type=str, 
        default="http://localhost:8000", 
        help="Base URL for the Mistral Inference Server"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="mistralai/mistral-7b",
        help="Model ID to use for inference"
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["chat", "completion"], 
        default="chat",
        help="API mode to use (chat or completion)"
    )
    
    parser.add_argument(
        "--stream", 
        action="store_true", 
        help="Stream the response token by token"
    )
    
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Sampling temperature (0.0 to 1.0)"
    )
    
    parser.add_argument(
        "--prompt", 
        type=str, 
        help="Prompt for completion mode"
    )
    
    parser.add_argument(
        "--system", 
        type=str, 
        default="You are a helpful AI assistant.",
        help="System message for chat mode"
    )
    
    parser.add_argument(
        "--user", 
        type=str, 
        help="User message for chat mode"
    )
    
    args = parser.parse_args()
    
    client = MistralClient(base_url=args.base_url)
    
    try:
        if args.mode == "chat":
            if not args.user:
                print("Error: --user message is required for chat mode")
                return
                
            messages = [
                {"role": "system", "content": args.system},
                {"role": "user", "content": args.user}
            ]
            
            if args.stream:
                print("Streaming response:")
                for chunk in client.chat_completion(
                    messages=messages,
                    model=args.model,
                    temperature=args.temperature,
                    stream=True
                ):
                    try:
                        content = chunk["choices"][0]["delta"].get("content", "")
                        print(content, end="", flush=True)
                    except:
                        pass
                print("\n")
            else:
                response = client.chat_completion(
                    messages=messages,
                    model=args.model,
                    temperature=args.temperature
                )
                print(response["choices"][0]["message"]["content"])
                
        elif args.mode == "completion":
            if not args.prompt:
                print("Error: --prompt is required for completion mode")
                return
                
            if args.stream:
                print("Streaming response:")
                for chunk in client.completion(
                    prompt=args.prompt,
                    model=args.model,
                    temperature=args.temperature,
                    stream=True
                ):
                    try:
                        content = chunk["choices"][0]["text"]
                        print(content, end="", flush=True)
                    except:
                        pass
                print("\n")
            else:
                response = client.completion(
                    prompt=args.prompt,
                    model=args.model,
                    temperature=args.temperature
                )
                print(response["choices"][0]["text"])
                
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
