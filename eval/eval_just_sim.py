import json
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from collections import defaultdict

def convert_to_serializable(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(x) for x in obj]
    return obj

def load_and_preprocess_data(json_file_path):
    """Load and preprocess data, ensuring valid responses exist."""
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    processed_data = []

    for entry in data:
        expected_response = entry.get("expected_output", "")
        if not isinstance(expected_response, str) or not expected_response.strip():
            continue

        try:
            model_responses = json.loads(entry.get("model_responses", "[]"))
        except (json.JSONDecodeError, TypeError):
            model_responses = entry.get("model_responses", [])
        
        if not isinstance(model_responses, list):
            continue

        checkpoint_data = []
        for model_response in model_responses:
            if isinstance(model_response, dict):
                checkpoint_data.append({
                    'model': model_response.get('model', 'unknown'),
                    'response': model_response.get('response', '')
                })
            else:
                checkpoint_data.append({
                    'model': 'unknown',
                    'response': str(model_response)
                })

        processed_data.append({
            'expected_response': expected_response.strip(),
            'checkpoint_responses': checkpoint_data
        })

    return processed_data

def calculate_cosine_similarity(embedding1, embedding2):
    return float(1 - cosine(embedding1, embedding2))  # Explicitly convert to Python float

def analyze_response_similarity(data):
    """Analyze similarity scores between expected and model responses across checkpoints"""
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    results = defaultdict(list)
    
    if not data:
        return results
    
    checkpoint_names = set()
    for entry in data:
        for resp in entry['checkpoint_responses']:
            checkpoint_names.add(resp['model'])
    
    for checkpoint in checkpoint_names:
        pairs = []
        for entry in data:
            checkpoint_response = next(
                (r for r in entry['checkpoint_responses'] if r['model'] == checkpoint),
                None
            )
            if checkpoint_response and checkpoint_response['response'].strip():
                pairs.append({
                    'expected': entry['expected_response'],
                    'model': checkpoint_response['response']
                })
        
        if not pairs:
            continue
        
        expected_texts = [p['expected'] for p in pairs]
        model_texts = [p['model'] for p in pairs]
        
        batch_size = 1
        expected_embeddings = []
        model_embeddings = []
        for i in range(0, len(expected_texts), batch_size):
            expected_batch = expected_texts[i:i+batch_size]
            model_batch = model_texts[i:i+batch_size]
            print(expected_batch)
            print(model_batch)
            expected_embeddings.extend(model.encode(expected_batch))
            model_embeddings.extend(model.encode(model_batch))
        
        similarities = []
        for expected_emb, model_emb in zip(expected_embeddings, model_embeddings):
            similarities.append(calculate_cosine_similarity(expected_emb, model_emb))
        
        # Convert all numpy types to native Python types
        results[checkpoint] = convert_to_serializable({
            'num_samples': len(pairs),
            'average_similarity': np.mean(similarities),
            'median_similarity': np.median(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities),
            'std_dev': np.std(similarities),
            'all_similarities': similarities
        })
    
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python response_similarity_analysis.py <json_file_path>")
        sys.exit(1)
    
    data = load_and_preprocess_data(sys.argv[1])
    
    if not data:
        print("No valid response data found.")
        sys.exit(0)
    
    results = analyze_response_similarity(data)
    
    print("\nResponse Similarity Analysis Results:")
    for checkpoint, metrics in results.items():
        print(f"\nCheckpoint: {checkpoint}")
        print(f"  Samples: {metrics['num_samples']}")
        print(f"  Average similarity: {metrics['average_similarity']:.4f}")
        print(f"  Median similarity: {metrics['median_similarity']:.4f}")
        print(f"  Minimum similarity: {metrics['min_similarity']:.4f}")
        print(f"  Maximum similarity: {metrics['max_similarity']:.4f}")
        print(f"  Standard deviation: {metrics['std_dev']:.4f}")
    
    with open('response_similarity_results_gemma_trainedckpt1k.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nDetailed results saved to response_similarity_results.json")