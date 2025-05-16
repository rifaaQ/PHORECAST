import json
import re
import os

def extract_numeric_value(response):
    if isinstance(response, str):
        response = response.strip()
        try:
            if response.startswith(('{', '[')):
                parsed = json.loads(response)
                if isinstance(parsed, list) and parsed:
                    if isinstance(parsed[0], dict) and 'response' in parsed[0]:
                        response = parsed[0]['response']
                    else:
                        return None
                elif isinstance(parsed, dict) and 'response' in parsed:
                    response = parsed['response']
        except json.JSONDecodeError:
            pass 
    if isinstance(response, dict) and 'response' in response:
        response = response['response']

    response = str(response)

    match = re.search(r'(?i)(?:answer:|is)?\s*([-+]?\d*\.?\d+)', response)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

def extract_last_question(instruction):    
    markers = [
        "Please predict the response to the following question: ",
    ]
    
    for marker in markers:
        if marker in instruction:
            question_part = instruction.rsplit(marker, 1)[-1]
            question = question_part.split('"')[0].strip()
            if question.endswith(':'):
                question = question[:-1].strip()
            return question
    
    return instruction.split('\n')[-1].strip()

def evaluate_accuracy(json_file_path,threshold):
    
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    results = {
        'total_samples': 0,
        'model_accuracy': {},  # Will contain model-specific stats
        'question_types': {}   # Will contain question type stats with model breakdown
    }
    
    for entry in data:
        expected = extract_numeric_value(entry['expected_output'])
        if expected is None:
            continue
            
        results['total_samples'] += 1
        
        try:
            model_responses = json.loads(entry['model_responses'])
        except (json.JSONDecodeError, TypeError):
            model_responses = entry['model_responses']
            
        if not isinstance(model_responses, list):
            continue
            
        question = extract_last_question(entry['instruction'])
        
        # Categorize the question
        question_key = None
        if "how open are you" in question.lower():
            question_key = "How open"
        elif "how harmful" in question.lower():
            question_key = "How harmful"
        elif any(f"feel {emotion}" in question.lower() for emotion in 
                ["worried", "angry", "sad", "afraid", "guilty", "disgusted", "ashamed", "hopeful"]):
            question_key = "Emotion rating"
        elif "more concerned" in question.lower():
            question_key = "Concern rating"
        elif "motivates me" in question.lower():
            question_key = "Motivation rating"
        else:
            question_key = "Other"
        

        if question_key not in results['question_types']:
            results['question_types'][question_key] = {
                'total': 0,
                'models': {}, 
                'questions': {}  
            }
        
        if question not in results['question_types'][question_key]['questions']:
            results['question_types'][question_key]['questions'][question] = {
                'total': 0,
                'models': {} 
            }
        
        results['question_types'][question_key]['total'] += 1
        results['question_types'][question_key]['questions'][question]['total'] += 1
        
        for model_response in model_responses:
            if isinstance(model_response, dict):
                model_name = model_response.get('model', 'unknown')
                response = model_response.get('response', '')
            else:
                model_name = 'unknown'
                response = str(model_response)
                
            predicted = extract_numeric_value(response)
            print(predicted)
            print(expected)
            # Initialize model tracking in global stats
            if model_name not in results['model_accuracy']:
                results['model_accuracy'][model_name] = {
                    'total': 0,
                    'correct': 0,
                    'by_question_type': {}  # Model's performance by question type
                }
            
            # Initialize model tracking for this question type
            if model_name not in results['question_types'][question_key]['models']:
                results['question_types'][question_key]['models'][model_name] = {
                    'total': 0,
                    'correct': 0
                }
            
            # Initialize model tracking for this specific question
            if model_name not in results['question_types'][question_key]['questions'][question]['models']:
                results['question_types'][question_key]['questions'][question]['models'][model_name] = {
                    'total': 0,
                    'correct': 0
                }
            
            results['model_accuracy'][model_name]['total'] += 1
            results['question_types'][question_key]['models'][model_name]['total'] += 1
            results['question_types'][question_key]['questions'][question]['models'][model_name]['total'] += 1
            
            if predicted is not None and abs(predicted - expected) < threshold:
                results['model_accuracy'][model_name]['correct'] += 1
                results['question_types'][question_key]['models'][model_name]['correct'] += 1
                results['question_types'][question_key]['questions'][question]['models'][model_name]['correct'] += 1
    
    # Calculate all accuracies
    # Global model accuracies
    for model in results['model_accuracy']:
        if results['model_accuracy'][model]['total'] > 0:
            results['model_accuracy'][model]['accuracy'] = (
                results['model_accuracy'][model]['correct'] / 
                results['model_accuracy'][model]['total']
            ) * 100
        else:
            results['model_accuracy'][model]['accuracy'] = 0
    
    # Question type accuracies
    for q_type in results['question_types']:
        # Calculate per-model accuracies for this question type
        for model in results['question_types'][q_type]['models']:
            stats = results['question_types'][q_type]['models'][model]
            if stats['total'] > 0:
                stats['accuracy'] = (stats['correct'] / stats['total']) * 100
            else:
                stats['accuracy'] = 0
        
        # Calculate per-model accuracies for each question in this type
        for question in results['question_types'][q_type]['questions']:
            for model in results['question_types'][q_type]['questions'][question]['models']:
                stats = results['question_types'][q_type]['questions'][question]['models'][model]
                if stats['total'] > 0:
                    stats['accuracy'] = (stats['correct'] / stats['total']) * 100
                else:
                    stats['accuracy'] = 0
    
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <json_file_path> <threshold>")
        sys.exit(1)

    json_file = sys.argv[1]
    try:
        threshold = float(sys.argv[2])
    except ValueError:
        print("Error: Threshold must be a numeric value.")
        sys.exit(1)


    results = evaluate_accuracy(json_file, threshold)

    for model, stats in results['model_accuracy'].items():
        print(f"{model}: {stats['accuracy']:.2f}% accuracy ({stats['correct']}/{stats['total']})")

    print("\nPerformance by Question Type:")
    for q_type, type_stats in results['question_types'].items():
        print(f"\n{q_type} (Total: {type_stats['total']})")
        print("Models:")
        for model, stats in type_stats['models'].items():
            print(f"  {model}: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")

        print("\nIndividual Questions:")
        for question, q_stats in type_stats['questions'].items():
            print(f"  {question} (Total: {q_stats['total']})")
            for model, stats in q_stats['models'].items():
                print(f"    {model}: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")

    savep = f"final_evaluation_results_{json_file.replace('.json', '')}_thres{threshold}.json"
    os.makedirs(os.path.dirname(savep), exist_ok=True)
    with open(savep, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {savep}")
    
    