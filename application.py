import os
import requests
import json
import hashlib
import re
import time
import uuid
import threading
from urllib.parse import urlparse
from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup
import asyncio
import platform

if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global stores
EMBEDDED_SITEMAPS = set()
document_cache = {}
progress_store = {}

def validate_url(value):
    try:
        result = urlparse(value)
        if all([result.scheme, result.netloc]):
            return value
        return False
    except ValueError:
        return False

def validate_sitemap(url):
    if not validate_url(url):
        return False
    if "sitemap" in url or url.endswith("xml"):
        return True
    return False

def load_sitemap(sitemap_url):
    """Load and parse sitemap directly with proper headers"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/xml, text/xml'
    }
    
    try:
        response = requests.get(sitemap_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Parse the sitemap
        soup = BeautifulSoup(response.content, 'xml')
        
        # Find all URLs in sitemap
        urls = []
        for loc in soup.find_all('loc'):
            url = loc.get_text().strip()
            if validate_url(url):
                urls.append(url)
        
        if not urls:
            # If no <loc> tags found, try to find sitemapindex
            for sitemap in soup.find_all('sitemap'):
                loc = sitemap.find('loc')
                if loc:
                    nested_url = loc.get_text().strip()
                    if validate_url(nested_url):
                        urls.extend(load_sitemap(nested_url))
        
        return list(set(urls))  # Remove duplicates
    except Exception as e:
        raise RuntimeError(f"Failed to load sitemap: {str(e)}")

def extract_page_content(url):
    """Extract text content from a webpage with proper headers"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Extract text content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            element.decompose()
        
        # Get text and clean
        text = soup.get_text()
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        print(f"Error extracting content from {url}: {str(e)}")
        return ""

def split_text_into_chunks(text, max_chunk_size=500):
    """Split text into chunks of specified size"""
    if not text:
        return []
    
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def get_embeddings(texts, hf_token):
    """Get embeddings using Hugging Face Inference API - CORRECTED"""
    # CORRECTED ENDPOINT
    url = f"https://router.huggingface.co/hf-inference/models/BAAI/bge-large-en-v1.5/pipeline/feature-extraction"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    # Send in batches to avoid large requests
    all_embeddings = []
    batch_size = 5
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        # CORRECTED PAYLOAD FORMAT
        payload = {
            "inputs": batch,
            "options": {"wait_for_model": True}
        }
        
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=300
            )
            
            if response.status_code != 200:
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', response.text[:200])
                except:
                    error_msg = response.text[:200]
                raise RuntimeError(f"Embedding API error ({response.status_code}): {error_msg}")
                
            # CORRECTED RESPONSE HANDLING
            batch_embeddings = response.json()
            if isinstance(batch_embeddings, list):
                all_embeddings.extend(batch_embeddings)
            else:
                raise RuntimeError(f"Unexpected response format: {type(batch_embeddings)}")
                
        except Exception as e:
            raise RuntimeError(f"Embedding batch failed: {str(e)}")
    
    return all_embeddings

def process_sitemap(task_id, sitemap_url, hf_token):
    """Process sitemap in a background thread and update progress"""
    try:
        # Create a unique hash for this sitemap
        sitemap_hash = hashlib.sha256(f"{sitemap_url}".encode()).hexdigest()
        
        # Check if we have it in cache
        if sitemap_hash in document_cache:
            progress_store[task_id] = {
                'progress': 100,
                'message': "Using cached embeddings",
                'complete': True,
                'error': None
            }
            return
        
        # 1. Load sitemap
        progress_store[task_id] = {
            'progress': 0,
            'message': "Loading sitemap...",
            'complete': False,
            'error': None
        }
        
        urls = load_sitemap(sitemap_url)
        if not urls:
            raise RuntimeError("No valid URLs found in sitemap")
        
        # 2. Extract and split content (limit to 10 URLs for stability)
        chunks = []
        total_urls = min(10, len(urls))
        
        for i, url in enumerate(urls[:10]):
            # Update progress
            progress = int((i / total_urls) * 70)
            progress_store[task_id] = {
                'progress': progress,
                'message': f"Processing URL {i+1}/{total_urls}",
                'complete': False,
                'error': None
            }
            
            content = extract_page_content(url)
            if content:
                url_chunks = split_text_into_chunks(content)
                for chunk in url_chunks:
                    chunks.append({
                        "url": url,
                        "content": chunk
                    })
            time.sleep(0.5)  # Be polite with requests
        
        if not chunks:
            raise RuntimeError("No content extracted from URLs")
        
        # Extract just the content for embeddings
        chunk_texts = [chunk["content"] for chunk in chunks]
        
        # 3. Create embeddings
        progress_store[task_id] = {
            'progress': 80,
            'message': "Generating embeddings...",
            'complete': False,
            'error': None
        }
        embeddings = get_embeddings(chunk_texts, hf_token)
        
        # Cache results
        document_cache[sitemap_hash] = {
            "chunks": chunks,
            "embeddings": embeddings,
            "chunk_texts": chunk_texts
        }
        EMBEDDED_SITEMAPS.add(sitemap_url)
        
        # Mark as complete
        progress_store[task_id] = {
            'progress': 100,
            'message': "Done!",
            'complete': True,
            'error': None
        }
        
    except Exception as e:
        progress_store[task_id] = {
            'progress': 0,
            'message': "Error occurred",
            'complete': True,
            'error': str(e)
        }

def get_query_embedding(query, hf_token):
    """Get embedding for a single query - CORRECTED"""
    # CORRECTED ENDPOINT
    url = f"https://router.huggingface.co/hf-inference/models/BAAI/bge-large-en-v1.5/pipeline/feature-extraction"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    # CORRECTED PAYLOAD FORMAT
    payload = {
        "inputs": query,
        "options": {"wait_for_model": True}
    }
    
    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            try:
                error_data = response.json()
                error_msg = error_data.get('error', response.text[:200])
            except:
                error_msg = response.text[:200]
            raise RuntimeError(f"Query embedding failed ({response.status_code}): {error_msg}")
            
        # CORRECTED RESPONSE HANDLING
        embedding = response.json()
        if isinstance(embedding, list) and isinstance(embedding[0], list):
            return embedding[0]
        elif isinstance(embedding, list) and isinstance(embedding[0], (int, float)):
            return embedding
        else:
            raise RuntimeError(f"Unexpected response format: {type(embedding)}")
            
    except Exception as e:
        raise RuntimeError(f"Query embedding failed: {str(e)}") from e

def get_most_relevant_chunks(query, document_data, hf_token):
    """Get most relevant chunks for a query"""
    # Get query embedding
    query_embedding = get_query_embedding(query, hf_token)
    
    # Calculate similarity (cosine similarity)
    embeddings = document_data["embeddings"]
    
    if not embeddings or not query_embedding:
        return []
    
    similarities = []
    for i, emb in enumerate(embeddings):
        if not emb:
            continue
            
        # Calculate dot product
        try:
            dot_product = sum(q * e for q, e in zip(query_embedding, emb))
            
            # Calculate magnitudes
            q_magnitude = sum(q * q for q in query_embedding) ** 0.5
            e_magnitude = sum(e * e for e in emb) ** 0.5
            
            # Avoid division by zero
            if q_magnitude * e_magnitude == 0:
                similarity = 0
            else:
                similarity = dot_product / (q_magnitude * e_magnitude)
                
            similarities.append((similarity, i))
        except Exception as e:
            print(f"Error calculating similarity: {str(e)}")
            continue
    
    # Get top 5 most relevant chunks
    similarities.sort(key=lambda x: x[0], reverse=True)
    top_indices = [i for _, i in similarities[:5]]
    return [document_data["chunk_texts"][i] for i in top_indices]

def get_chat_response(query, context, hf_token):
    """Get response using HF Router chat-completions with DeepSeek only and sanitize output"""

    def _clean_deepseek_output(text: str) -> str:
        # Remove <think>...</think> blocks
        import re
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

        # If the model wrote "Final Response:" or "Answer:", keep only what's after it
        for marker in ["Final Response:", "Answer:", "Final answer:", "Final:" ]:
            idx = text.lower().find(marker.lower())
            if idx != -1:
                text = text[idx + len(marker):]
                break

        # Strip common prefixes and extra whitespace
        text = re.sub(r"^\s*Assistant:\s*", "", text, flags=re.IGNORECASE)
        return text.strip()

    # Build user message (reuse your existing context formatting)
    if context:
        context_str = "\n\n".join([f"[Context {i+1}]: {c}" for i, c in enumerate(context)])
        user_content = f"Use the following context to answer the question:\n\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
    else:
        user_content = f"Question: {query}\n\nAnswer:"

    api_url = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-ai/DeepSeek-R1-0528:fireworks-ai",  # force DeepSeek
        "messages": [{"role": "user", "content": user_content}],
        "temperature": 0.7
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=120)
        if response.status_code != 200:
            try:
                err = response.json()
                err_msg = err.get("error", {}).get("message") or err.get("message") or str(err)
            except Exception:
                err_msg = response.text[:200]
            raise RuntimeError(f"Chat API error ({response.status_code}): {err_msg}")

        data = response.json()

        # Primary path: choices[0].message.content
        content = None
        if isinstance(data, dict) and data.get("choices"):
            msg = data["choices"][0].get("message", {})
            content = msg.get("content")

            # Some providers may put the final answer in choices[0].text
            if not content:
                content = data["choices"][0].get("text")

        if not isinstance(content, str) or not content.strip():
            return "No response generated"

        # ✨ Ensure only the response text is returned
        return _clean_deepseek_output(content)

    except Exception as e:
        raise RuntimeError(f"Chat API error: {str(e)}") from e

# API Endpoints
@app.route('/api/process', methods=['POST'])
def api_process():
    data = request.json
    hf_token = data.get('hf_token')
    sitemap_url = data.get('sitemap_url')
    
    if not validate_sitemap(sitemap_url):
        return jsonify({"error": "Invalid sitemap URL"}), 400
    
    # Create task ID
    task_id = str(uuid.uuid4())
    
    # Initialize progress
    progress_store[task_id] = {
        'progress': 0,
        'message': "Starting...",
        'complete': False,
        'error': None
    }
    
    # Start processing in background thread
    thread = threading.Thread(
        target=process_sitemap,
        args=(task_id, sitemap_url, hf_token)
    )
    thread.start()
    
    return jsonify({"task_id": task_id})

@app.route('/api/progress/<task_id>', methods=['GET'])
def api_progress(task_id):
    if task_id in progress_store:
        return jsonify(progress_store[task_id])
    else:
        return jsonify({
            'progress': 0,
            'message': "Task not found",
            'complete': False,
            'error': None
        }), 404

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.json
    hf_token = data.get('hf_token')
    sitemap_url = data.get('sitemap_url')
    question = data.get('question')
    
    if not all([hf_token, sitemap_url, question]):
        return jsonify({"error": "Missing required parameters"}), 400
    
    # Get document chunks
    sitemap_hash = hashlib.sha256(f"{sitemap_url}".encode()).hexdigest()
    if sitemap_hash not in document_cache:
        return jsonify({"error": "Sitemap not processed"}), 400
    
    try:
        document_data = document_cache[sitemap_hash]
        context = get_most_relevant_chunks(question, document_data, hf_token)
        response = get_chat_response(question, context, hf_token)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def api_health():
    return jsonify({"status": "ok", "version": "1.0"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)