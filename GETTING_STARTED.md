# Getting Started with Mazag Backend 🚀

## Overview

Mazag is a comprehensive mental health chatbot backend that combines:
- **RAG** (Retrieval-Augmented Generation) for context-aware responses
- **Guardrails** for safety and ethical AI behavior  
- **openai/gpt-oss-120b** for conversational intelligence
- **Sentiment Analysis** for understanding user emotional state
- **Therapist Matching** using advanced similarity algorithms

## Installation

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

This will install:
- `openai` - For Gemini API via OpenAI compatibility
- `sentence-transformers` - For text embeddings
- `faiss-cpu` - For vector similarity search
- `scikit-learn` - For ML utilities
- `numpy` - For numerical operations

### 2. Get Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key

### 3. Test Installation

Run the component tests:

```bash
python test_components.py
```

This will verify all components are working correctly.

## Quick Start (5 minutes)

### Minimal Example

```python
from mazag_prototype import create_mazag_engine

# 1. Initialize
engine = create_mazag_engine(api_key="YOUR_API_KEY")

# 2. Chat
result = engine.process_message("I'm feeling anxious")
print(result["response"])
```

### With Knowledge Base

```python
from mazag_prototype import create_mazag_engine

# Initialize
engine = create_mazag_engine(api_key="YOUR_API_KEY")

# Add mental health knowledge
engine.add_knowledge("""
Anxiety is characterized by excessive worry.
CBT helps by identifying cognitive distortions.
""")

# Process with RAG
result = engine.process_message(
    "How can I deal with worrying thoughts?",
    use_rag=True,
    analyze_sentiment=True
)

print(f"Response: {result['response']}")
print(f"Sentiment: {result['analysis']['sentiment']}")
print(f"Retrieved context: {len(result['retrieved_context'])} chunks")
```

### Complete Example with Recommendations

```python
import json
from mazag_prototype import create_mazag_engine

# Initialize
engine = create_mazag_engine(api_key="YOUR_API_KEY")

# Load therapist data
with open('vezeeta_psychiatrists.json', 'r') as f:
    therapists = json.load(f)

# Get recommendations
recommendations = engine.get_therapist_recommendations(
    patient_concerns=["anxiety", "depression"],
    therapist_profiles=therapists[:100],  # Use subset
    preferred_language="Arabic"
)

# Show top 3
for rec in recommendations[:3]:
    print(f"{rec['therapist']['name']}: {rec['score']:.2f}")
    print(f"  Reasons: {', '.join(rec['match_reasons'])}")
```

## Running the Demo

### Option 1: Python Script

```bash
python quick_start.py
```

Edit `quick_start.py` to add your API key first.

### Option 2: Jupyter Notebook

```bash
jupyter notebook mazag_prototype_demo.ipynb
```

The notebook includes:
- ✅ Step-by-step component testing
- ✅ Full integration examples
- ✅ Configuration experiments
- ✅ Recommendation demos
- ✅ Save/load examples

## Architecture Overview

```
User Input
    ↓
[Guardrails] ← Input filtering
    ↓
[Analyzer] ← Sentiment/emotion detection
    ↓
[RAG] ← Retrieve relevant context
    ↓
[Chatbot] ← Generate response with Gemini
    ↓
[Guardrails] ← Output verification
    ↓
Response
```

## Configuration Options

### Basic Configuration

```python
from mazag_prototype import MazagConfig, MazagEngine

config = MazagConfig(
    api_key="YOUR_KEY",
    
    # RAG settings
    chunking_strategy="sentence",  # or "fixed", "semantic"
    chunk_size=512,
    chunk_overlap=50,
    embedding_model="sentence-transformer",
    rag_top_k=3,
    
    # AI settings
    temperature=0.7,  # Higher = more creative
    max_tokens=500,
    
    # Analysis
    analysis_method="lexicon",  # or "transformers", "gemini"
    
    # Safety
    strict_mode=False
)

engine = MazagEngine(config)
```

### Experiment with Different Settings

```python
# Try different similarity metrics
for metric in ["cosine", "l2", "ip"]:
    engine = create_mazag_engine(
        api_key=API_KEY,
        similarity_metric=metric
    )
    # Test retrieval...

# Try different context sizes
for k in [1, 3, 5, 10]:
    engine = create_mazag_engine(
        api_key=API_KEY,
        rag_top_k=k
    )
    # Compare responses...
```

## Using Individual Components

### RAG Components

```python
from mazag_prototype.rag import (
    DocumentChunker, 
    create_embedder, 
    FAISSVectorStore
)

# Chunk documents
chunker = DocumentChunker(chunk_size=512, strategy="sentence")
chunks = chunker.chunk_text("Your document...")

# Generate embeddings
embedder = create_embedder("sentence-transformer")
embeddings = embedder.embed([c.text for c in chunks])

# Store & retrieve
store = FAISSVectorStore(embedding_dim=384, metric="cosine")
store.add_batch(
    chunk_ids=[c.chunk_id for c in chunks],
    texts=[c.text for c in chunks],
    embeddings=embeddings
)

# Search
query_emb = embedder.embed("anxiety symptoms")
results = store.search(query_emb, k=5)
```

### Guardrails

```python
from mazag_prototype.ai import GuardrailsSystem

guardrails = GuardrailsSystem(strict_mode=False)

# Check input
result = guardrails.check_input("I want to hurt myself")

if not result["allowed"]:
    if result["critical_issue"]:
        # Handle crisis
        print(guardrails.get_crisis_response())
    else:
        # Handle other blocks
        print("Input blocked")
```

### Chatbot

```python
from mazag_prototype.ai import MazagChatbot

chatbot = MazagChatbot(api_key="YOUR_KEY")

# Single response
response = chatbot.generate_response("I feel sad")

# With context
response = chatbot.generate_response(
    "I feel sad",
    context="Depression is characterized by persistent sadness..."
)

# Conversation
chatbot.generate_response("Hi, I need help")
chatbot.generate_response("I'm feeling anxious")
# History is maintained automatically
```

### Analysis

```python
from mazag_prototype.ai import create_analyzer

analyzer = create_analyzer("lexicon")
analysis = analyzer.analyze("I feel hopeless and alone")

print(f"Sentiment: {analysis.sentiment}")
print(f"Emotions: {analysis.emotions}")
print(f"Tone: {analysis.tone}")
print(f"Risk: {analysis.risk_level}")
print(f"Indicators: {analysis.indicators}")
```

### Recommendations

```python
from mazag_prototype.recommend import recommend_therapists

recommendations = recommend_therapists(
    patient_description="I have anxiety and need CBT therapy",
    therapist_profiles=therapist_data,
    top_k=5
)

for rec in recommendations:
    print(rec['therapist']['name'])
    print(f"Score: {rec['score']}")
    print(f"Reasons: {rec['reasons']}")
```

## Common Use Cases

### 1. Mental Health Chatbot

```python
engine = create_mazag_engine(api_key=API_KEY)

# Add CBT knowledge base
with open('cbt_guide.txt', 'r') as f:
    engine.add_knowledge(f.read())

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    
    result = engine.process_message(user_input, use_rag=True)
    print(f"Mazag: {result['response']}")
```

### 2. Therapist Matching Service

```python
# Get patient input
patient_concerns = ["anxiety", "panic attacks"]
preferred_language = "Arabic"

# Get recommendations
recs = engine.get_therapist_recommendations(
    patient_concerns=patient_concerns,
    therapist_profiles=all_therapists,
    preferred_language=preferred_language,
    preferred_approach=["CBT"]
)

# Display to user
for rec in recs[:5]:
    print(f"{rec['therapist']['name']}")
    print(f"  Match: {rec['score']*100:.0f}%")
    print(f"  Why: {', '.join(rec['match_reasons'])}")
```

### 3. Conversation Analysis Dashboard

```python
# Analyze multiple conversations
conversations = load_conversations()

for conv in conversations:
    for message in conv['messages']:
        analysis = engine.analyzer.analyze(message['text'])
        
        # Track metrics
        if analysis.risk_level == "high":
            flag_for_review(conv['id'])
        
        # Store sentiment trends
        store_analysis(conv['id'], analysis)
```

## Troubleshooting

### Issue: Import errors

**Solution**: Make sure you're running from the `backend` directory:

```bash
cd backend
python quick_start.py
```

### Issue: API key errors

**Solution**: Verify your API key:

```python
import openai

client = openai.OpenAI(
    api_key="YOUR_KEY",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{"role": "user", "content": "test"}]
)
print(response.choices[0].message.content)
```

### Issue: Slow embeddings

**Solution**: Use GPU version of FAISS:

```bash
pip uninstall faiss-cpu
pip install faiss-gpu
```

Or use lighter embedding model:

```python
embedder = create_embedder("tfidf")
```

### Issue: Memory errors with large datasets

**Solution**: Process in batches:

```python
batch_size = 100
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    for doc in batch:
        engine.add_knowledge(doc)
```

## Performance Tips

1. **Use FAISS for large datasets** (>1000 chunks)
2. **Batch operations** when adding multiple documents
3. **Save/load vector stores** instead of rebuilding
4. **Adjust chunk_size** based on your use case
5. **Use lexicon analyzer** for speed (vs transformers)

## Next Steps

1. ✅ **Integrate with frontend**: See `/app` directory
2. ✅ **Deploy as API**: Create FastAPI endpoints
3. ✅ **Add authentication**: User sessions and API keys
4. ✅ **Monitor performance**: Track response times and quality
5. ✅ **Fine-tune**: Create custom embeddings for mental health

## Resources

- **Demo Notebook**: `mazag_prototype_demo.ipynb`
- **Full README**: `README.md`
- **Component Tests**: `test_components.py`
- **Quick Start**: `quick_start.py`
- **Project Docs**: `../documents/Graduation_Project_Mazag.pdf`

## Support

For issues or questions:
1. Check the demo notebook for examples
2. Review component tests
3. Consult the full README
4. Check project documentation

---

**Ready to build something amazing! 🚀**

