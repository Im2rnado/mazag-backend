import sys
import os
import json
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.services.chatbot import generate_response

async def run_tests():
    test_cases_path = os.path.join(os.path.dirname(__file__), "test_cases.json")
    results_path = os.path.join(os.path.dirname(__file__), "test_results.json")
    
    if not os.path.exists(test_cases_path):
        print(f"Error: {test_cases_path} not found.")
        return

    with open(test_cases_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
        
    results = []
    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
            
    completed_questions = {r['question'] for r in results}
    remaining_cases = [tc for tc in test_cases if tc['question'] not in completed_questions]
    
    print(f"Loaded {len(test_cases)} total test cases.")
    print(f"Resuming with {len(remaining_cases)} remaining cases...")
    
    for i, tc in enumerate(remaining_cases):
        print(f"Testing {len(results)+1}/{len(test_cases)} [{tc['category']}]...")
        retries = 3
        base_delay = 5
        
        for attempt in range(retries):
            try:
                response_text, analysis = await generate_response(
                    message=tc['question'],
                    history=[],
                    context=None
                )
                
                result = {
                    "category": tc["category"],
                    "question": tc["question"],
                    "response": response_text,
                    "analysis": analysis,
                    "guardrail_triggered": analysis.get("guardrail") is not None
                }
                results.append(result)
                break
                
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "rate limit" in error_str or "timeout" in error_str:
                    delay = base_delay * (2 ** attempt)
                    print(f"Rate limited. Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    print(f"Error on test {len(results)+1}: {e}")
                    result = {
                        "category": tc["category"],
                        "question": tc["question"],
                        "response": f"ERROR: {e}",
                        "analysis": {},
                        "guardrail_triggered": False
                    }
                    results.append(result)
                    break
        else:
            # All retries failed
            print(f"Failed to process after {retries} attempts due to rate limits.")
            break
            
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        await asyncio.sleep(2)
            
    print(f"\nSuccessfully tested {len(results)} cases.")
    print(f"Saved to {results_path}")

if __name__ == "__main__":
    asyncio.run(run_tests())
