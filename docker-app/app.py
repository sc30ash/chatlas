import os
import requests
import json
import hashlib
import re
import time
import uuid
import threading
import boto3
import gzip
import base64
from decimal import Decimal
from urllib.parse import urlparse
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from bs4 import BeautifulSoup
from botocore.exceptions import ClientError

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# S3 Configuration
S3_BUCKET_NAME = 'sitemap-processor-cache-880422432103'

# Initialize AWS clients
try:
    dynamodb = boto3.resource('dynamodb', region_name='us-east-2')
    progress_table = dynamodb.Table('sitemap-progress')
    cache_table = dynamodb.Table('sitemap-cache')
    print("DynamoDB connected successfully")
except Exception as e:
    print(f"DynamoDB connection failed: {e}")
    dynamodb = None

try:
    s3_client = boto3.client('s3', region_name='us-east-2')
    print("S3 client initialized successfully")
except Exception as e:
    print(f"S3 client initialization failed: {e}")
    s3_client = None

# Global stores (fallback for local development)
EMBEDDED_SITEMAPS = set()
local_document_cache = {}
local_progress_store = {}

# Helper functions for DynamoDB float conversion
def convert_floats_to_decimal(obj):
    """Recursively convert all float values to Decimal for DynamoDB compatibility"""
    if isinstance(obj, float):
        return Decimal(str(obj))
    elif isinstance(obj, dict):
        return {k: convert_floats_to_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats_to_decimal(i) for i in obj]
    else:
        return obj

class DecimalEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle Decimal objects"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)

# S3 Integration Functions
def store_large_data_in_s3(sitemap_hash, cache_data):
    """Store large data in S3 with compression"""
    if not s3_client:
        print("S3 client not available, falling back to local storage")
        return None
        
    try:
        # Convert data to JSON and compress
        json_data = json.dumps(cache_data, cls=DecimalEncoder)
        compressed = gzip.compress(json_data.encode('utf-8'))
        
        # Generate S3 key
        s3_key = f"sitemap-cache/{sitemap_hash}.json.gz"
        
        # Store in S3
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=compressed,
            ContentType='application/gzip',
            Metadata={
                'sitemap-hash': sitemap_hash,
                'created-at': str(int(time.time()))
            }
        )
        
        print(f"Data stored in S3: s3://{S3_BUCKET_NAME}/{s3_key}")
        return s3_key
        
    except Exception as e:
        print(f"S3 storage error: {e}")
        return None

def get_large_data_from_s3(s3_key):
    """Retrieve and decompress large data from S3"""
    if not s3_client:
        print("S3 client not available")
        return None
        
    try:
        print(f"Retrieving data from S3: s3://{S3_BUCKET_NAME}/{s3_key}")
        
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        compressed_data = response['Body'].read()
        
        # Decompress and parse JSON
        json_data = gzip.decompress(compressed_data).decode('utf-8')
        data = json.loads(json_data)
        
        print(f"Successfully retrieved {len(json_data)} bytes from S3")
        return data
        
    except Exception as e:
        print(f"S3 retrieval error: {e}")
        return None

def delete_s3_data(s3_key):
    """Delete data from S3 (optional cleanup)"""
    if not s3_client:
        return False
        
    try:
        s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        print(f"Deleted S3 object: {s3_key}")
        return True
    except Exception as e:
        print(f"S3 deletion error: {e}")
        return False

# DynamoDB helper functions with S3 integration
def get_progress(task_id):
    """Get progress from DynamoDB"""
    if not dynamodb:
        return local_progress_store.get(task_id)
    
    try:
        response = progress_table.get_item(Key={'task_id': task_id})
        if 'Item' in response:
            return response['Item']['data']
        return None
    except ClientError as e:
        print(f"DynamoDB get_progress error: {e}")
        return None

def set_progress(task_id, progress_data):
    """Set progress in DynamoDB with float conversion"""
    if not dynamodb:
        local_progress_store[task_id] = progress_data
        return
    
    try:
        # Convert floats to decimals before storing
        decimal_data = convert_floats_to_decimal(progress_data)
        
        progress_table.put_item(
            Item={
                'task_id': task_id,
                'data': decimal_data,
                'ttl': int(time.time()) + 3600  # 1 hour TTL
            }
        )
        print(f"Progress saved for task {task_id}: {progress_data}")
    except ClientError as e:
        print(f"DynamoDB set_progress error: {e}")

def get_document_cache(sitemap_hash):
    """Get data from S3 using DynamoDB reference"""
    if not dynamodb:
        return local_document_cache.get(sitemap_hash)
    
    try:
        response = cache_table.get_item(Key={'sitemap_hash': sitemap_hash})
        if 'Item' in response:
            item = response['Item']
            
            if item.get('storage_type') == 's3':
                s3_key = item.get('s3_key')
                if s3_key:
                    print(f"Retrieving data from S3 for hash {sitemap_hash}")
                    return get_large_data_from_s3(s3_key)
                else:
                    print("No S3 key found in DynamoDB item")
                    return None
            else:
                # Fallback for old format data stored directly in DynamoDB
                return item.get('data')
        else:
            print(f"No cache entry found for hash {sitemap_hash}")
            return None
            
    except ClientError as e:
        print(f"DynamoDB get_document_cache error: {e}")
        return local_document_cache.get(sitemap_hash)

def set_document_cache(sitemap_hash, cache_data):
    """Store reference to S3 object in DynamoDB"""
    if not dynamodb:
        local_document_cache[sitemap_hash] = cache_data
        return
    
    try:
        # Store large data in S3
        s3_key = store_large_data_in_s3(sitemap_hash, cache_data)
        if not s3_key:
            print("Failed to store in S3, falling back to local cache")
            local_document_cache[sitemap_hash] = cache_data
            return
        
        # Store only reference in DynamoDB
        decimal_metadata = convert_floats_to_decimal({
            'sitemap_hash': sitemap_hash,
            's3_key': s3_key,
            'storage_type': 's3',
            'bucket_name': S3_BUCKET_NAME,
            'created_at': int(time.time()),
            'data_size_kb': len(json.dumps(cache_data, cls=DecimalEncoder)) / 1024,
            'ttl': int(time.time()) + 86400  # 24 hour TTL
        })
        
        cache_table.put_item(Item=decimal_metadata)
        print(f"S3 reference saved in DynamoDB for hash {sitemap_hash}")
        
    except ClientError as e:
        print(f"DynamoDB set_document_cache error: {e}")
        # Fallback to local storage
        local_document_cache[sitemap_hash] = cache_data

# Cache cleanup function (optional)
def cleanup_expired_cache():
    """Clean up expired cache entries from both DynamoDB and S3"""
    if not dynamodb or not s3_client:
        return
        
    try:
        current_time = int(time.time())
        
        # Scan for expired items
        response = cache_table.scan(
            FilterExpression='#ttl < :current_time',
            ExpressionAttributeNames={'#ttl': 'ttl'},
            ExpressionAttributeValues={':current_time': current_time}
        )
        
        for item in response.get('Items', []):
            sitemap_hash = item.get('sitemap_hash')
            s3_key = item.get('s3_key')
            
            if s3_key:
                # Delete from S3
                delete_s3_data(s3_key)
            
            # Delete from DynamoDB
            cache_table.delete_item(Key={'sitemap_hash': sitemap_hash})
            print(f"Cleaned up expired cache for hash {sitemap_hash}")
            
    except Exception as e:
        print(f"Cache cleanup error: {e}")

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
        soup = BeautifulSoup(response.content, 'html.parser')
        
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
    """Split text into chunks of specified size with fixed regex"""
    if not text:
        return []
    
    # Fixed regex pattern - simple sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
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
    """Get embeddings using Hugging Face Inference API"""
    url = f"https://router.huggingface.co/hf-inference/models/BAAI/bge-large-en-v1.5/pipeline/feature-extraction"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    # Send in batches to avoid large requests
    all_embeddings = []
    batch_size = 5
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
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
        cached_data = get_document_cache(sitemap_hash)
        if cached_data:
            set_progress(task_id, {
                'progress': 100,
                'message': "Using cached embeddings",
                'complete': True,
                'error': None
            })
            return
        
        # 1. Load sitemap
        set_progress(task_id, {
            'progress': 0,
            'message': "Loading sitemap...",
            'complete': False,
            'error': None
        })
        
        urls = load_sitemap(sitemap_url)
        if not urls:
            raise RuntimeError("No valid URLs found in sitemap")
        
        # 2. Extract and split content (REDUCED to 3 URLs to stay under S3/DynamoDB limits)
        chunks = []
        total_urls = min(10, len(urls))  # Reduced from 10 to 3
        
        for i, url in enumerate(urls[:10]):  # Process only 3 URLs
            # Update progress
            progress = int((i / total_urls) * 70)
            set_progress(task_id, {
                'progress': progress,
                'message': f"Processing URL {i+1}/{total_urls}",
                'complete': False,
                'error': None
            })
            
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
        set_progress(task_id, {
            'progress': 80,
            'message': "Generating embeddings...",
            'complete': False,
            'error': None
        })
        embeddings = get_embeddings(chunk_texts, hf_token)
        
        # Cache results (now stored in S3 via set_document_cache)
        cache_data = {
            "chunks": chunks,
            "embeddings": embeddings,
            "chunk_texts": chunk_texts
        }
        set_document_cache(sitemap_hash, cache_data)
        EMBEDDED_SITEMAPS.add(sitemap_url)
        
        # Mark as complete
        set_progress(task_id, {
            'progress': 100,
            'message': "Done!",
            'complete': True,
            'error': None
        })
        
    except Exception as e:
        print(f"Error processing sitemap: {str(e)}")
        set_progress(task_id, {
            'progress': 0,
            'message': "Error occurred",
            'complete': True,
            'error': str(e)
        })

def get_query_embedding(query, hf_token):
    """Get embedding for a single query"""
    url = f"https://router.huggingface.co/hf-inference/models/BAAI/bge-large-en-v1.5/pipeline/feature-extraction"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
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
            
        embedding = response.json()
        
        # Handle different response formats from HF API
        if isinstance(embedding, list):
            # Check if it's a list of numbers (flat embedding vector)
            if len(embedding) > 0 and isinstance(embedding[0], (int, float)):
                print(f"Received flat embedding vector of length {len(embedding)}")
                return embedding
            # Check if it's a list of lists (batch response)
            elif len(embedding) > 0 and isinstance(embedding, list):
                print(f"Received nested embedding, taking first one")
                return embedding
        
        # If we get here, the format is unexpected
        raise RuntimeError(f"Unexpected embedding format: type={type(embedding)}, length={len(embedding) if hasattr(embedding, '__len__') else 'N/A'}")
            
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
    """Get response using HF Router chat-completions with DeepSeek"""

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
                content = data["choices"].get("text")

        if not isinstance(content, str) or not content.strip():
            return "No response generated"

        # ✨ Ensure only the response text is returned
        return _clean_deepseek_output(content)

    except Exception as e:
        raise RuntimeError(f"Chat API error: {str(e)}") from e

# API Endpoints
@app.route('/api/process', methods=['POST'])
def api_process():
    # Add debug logging
    print("=== DEBUG INFO ===")
    print(f"Content-Type: {request.content_type}")
    print(f"Request method: {request.method}")
    print(f"Request data: {request.get_data()}")
    print(f"Request JSON: {request.get_json(silent=True)}")
    print("==================")
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        hf_token = data.get('hf_token')
        sitemap_url = data.get('sitemap_url')
        
        print(f"Parsed - hf_token: {hf_token}, sitemap_url: {sitemap_url}")
        
        if not hf_token:
            return jsonify({"error": "Missing hf_token"}), 400
        if not validate_sitemap(sitemap_url):
            return jsonify({"error": "Invalid sitemap URL"}), 400
        
        # Create task ID
        task_id = str(uuid.uuid4())
        
        # Initialize progress
        set_progress(task_id, {
            'progress': 0,
            'message': "Starting...",
            'complete': False,
            'error': None
        })
        
        # Start processing in background thread
        thread = threading.Thread(
            target=process_sitemap,
            args=(task_id, sitemap_url, hf_token)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({"task_id": task_id})
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({"error": f"Request processing failed: {str(e)}"}), 400

@app.route('/api/progress/<task_id>', methods=['GET'])
def api_progress(task_id):
    progress_data = get_progress(task_id)
    if progress_data:
        # Use custom encoder to handle Decimals
        json_str = json.dumps(progress_data, cls=DecimalEncoder)
        return Response(json_str, mimetype='application/json')
    else:
        return jsonify({
            'progress': 0,
            'message': "Task not found",
            'complete': False,
            'error': None
        }), 404

@app.route('/api/chat', methods=['POST'])
def api_chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        hf_token = data.get('hf_token')
        sitemap_url = data.get('sitemap_url')
        question = data.get('question')
        
        if not all([hf_token, sitemap_url, question]):
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Get document chunks
        sitemap_hash = hashlib.sha256(f"{sitemap_url}".encode()).hexdigest()
        document_data = get_document_cache(sitemap_hash)
        
        if not document_data:
            return jsonify({"error": "Sitemap not processed"}), 400
        
        context = get_most_relevant_chunks(question, document_data, hf_token)
        response = get_chat_response(question, context, hf_token)
        
        # Use custom encoder for response
        result = {"response": response}
        json_str = json.dumps(result, cls=DecimalEncoder)
        return Response(json_str, mimetype='application/json')
        
    except Exception as e:
        print(f"Chat error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Cache cleanup endpoint (optional)
@app.route('/api/cleanup-cache', methods=['POST'])
def api_cleanup_cache():
    """Manual cache cleanup endpoint"""
    try:
        cleanup_expired_cache()
        return jsonify({"message": "Cache cleanup completed"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response
    
@app.route('/api/health', methods=['GET'])
def api_health():
    return jsonify({"status": "ok", "version": "1.2-s3-integrated"})

@app.route('/<path:path>', methods=['OPTIONS'])
@app.route('/', methods=['OPTIONS'])
def handle_options(path=None):
    return '', 200

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 80))
    app.run(host='0.0.0.0', port=port, debug=False)
