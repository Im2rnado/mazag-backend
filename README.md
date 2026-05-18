# Mazag Backend Prototype

A comprehensive AI-powered mental health companion backend with **RAG (Retrieval-Augmented Generation)**, **Guardrails**, **Sentiment Analysis**, and **Therapist Recommendation** systems.

## Features

### 1. **RAG Pipeline**
- **Chunking**: Multiple strategies (fixed, sentence-based, semantic)
- **Embedding**: Support for Sentence Transformers, Gemini embeddings, TF-IDF
- **Vector Store**: FAISS-powered similarity search with multiple metrics
- **Retrieval**: Context-aware document retrieval for enhanced responses

### 2. **Guardrails System**
- **Input Filtering**: Blocks harmful content, off-topic queries, spam
- **Output Checking**: Ensures ethical, safe AI responses
- **Crisis Detection**: Identifies red flags (suicide, self-harm, violence)
- **Medical Advice Prevention**: Prevents inappropriate diagnosis/prescription

### 3. **AI Chatbot**
- **openai/gpt-oss-120b Integration**: Using OpenAI-compatible API
- **Mazag Persona**: Warm, empathetic, inquisitive mental health companion
- **CBT-Informed**: Includes cognitive behavioral therapy knowledge
- **Conversation Management**: Maintains context across sessions

### 4. **Sentiment & Emotion Analysis**
- **Multi-Method Support**: Lexicon-based, Transformers, text2emotion, Gemini-powered
- **Comprehensive Analysis**: Sentiment, emotions, tone, psychological indicators
- **Risk Assessment**: Automatic risk level detection (low/medium/high)
- **Cognitive Distortion Detection**: Identifies negative thought patterns

### 5. **Therapist Recommendation**
- **Ensemble Similarity Metrics**: Cosine, Euclidean, Dot Product, Manhattan, Jaccard
- **Multi-Factor Matching**: Specialties, languages, therapy approaches, embeddings
- **Explainable Recommendations**: Clear reasons for each match
- **Weighted Scoring**: Configurable importance weights for different factors

## Architecture

```
mazag_prototype/
├── rag/
│   ├── chunker.py              # Document chunking strategies
│   ├── embedder.py             # Text embedding (multiple models)
│   └── vector_store.py         # FAISS vector storage & retrieval
├── ai/
│   ├── guardrails.py           # Safety & filtering system
│   ├── chatbot.py              # Gemini-powered chatbot
│   └── analyzer.py             # Sentiment/emotion analysis
├── recommend/
│   └── recommender.py          # Therapist recommendation engine
└── main.py                     # Unified integration layer
```

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from mazag_prototype import create_mazag_engine

# Initialize engine
engine = create_mazag_engine(
    api_key="YOUR_GEMINI_API_KEY",
    embedding_model="sentence-transformer",
    analysis_method="lexicon"
)

# Add knowledge base
engine.add_knowledge("""
Anxiety is characterized by excessive worry and fear.
Common symptoms include restlessness and difficulty concentrating.
""")

# Process user message
result = engine.process_message(
    "I'm feeling really anxious lately",
    use_rag=True,
    analyze_sentiment=True
)

print(result["response"])
# Output: Empathetic, context-aware response from Mazag
```

### Complete Example with All Features

```python
# 1. Setup
from mazag_prototype import MazagEngine, MazagConfig

config = MazagConfig(
    api_key="YOUR_API_KEY",
    chunking_strategy="sentence",
    embedding_model="sentence-transformer",
    rag_top_k=3,
    temperature=0.7
)

engine = MazagEngine(config)

# 2. Add Mental Health Knowledge
knowledge_docs = [
    "CBT helps identify cognitive distortions like catastrophizing...",
    "Mindfulness reduces anxiety by staying present...",
    # ... more documents
]

for doc in knowledge_docs:
    engine.add_knowledge(doc, metadata={"source": "cbt_guide"})

# 3. Process Conversation
response = engine.process_message("I can't stop worrying about everything")

# 4. Get Therapist Recommendations
recommendations = engine.get_therapist_recommendations(
    patient_concerns=["anxiety", "worry"],
    therapist_profiles=therapist_data,
    preferred_language="English",
    preferred_approach=["CBT"]
)

for rec in recommendations[:3]:
    print(f"{rec['therapist']['name']}: {rec['score']:.2f}")
    print(f"Reasons: {', '.join(rec['match_reasons'])}")
```

## Demo Notebook

Check out `mazag_prototype_demo.ipynb` for a comprehensive walkthrough including:
- Component testing (RAG, Guardrails, Analysis)
- Full integration examples
- Configuration experiments
- Recommendation system demo
- Save/load vector stores

## Experiments & Configurations

### RAG Configuration

```python
# Experiment with different chunking strategies
strategies = ["fixed", "sentence", "semantic"]

# Compare embedding models
models = ["sentence-transformer", "tfidf", "gemini"]

# Test similarity metrics
metrics = ["cosine", "euclidean", "dot_product"]

# Vary context size
top_k_values = [1, 3, 5, 10]
```

### Recommendation Tuning

```python
# Adjust feature weights
recommender = TherapistRecommender(
    therapists=therapist_list,
    embedding_weight=0.4,      # Semantic similarity
    specialty_weight=0.3,      # Matching specializations
    language_weight=0.2,       # Language preference
    approach_weight=0.1        # Therapy approach match
)

# Try ensemble methods
ensemble_methods = ["weighted_average", "max", "min"]
```

## Testing Components Individually

### RAG Components

```python
from mazag_prototype.rag import DocumentChunker, create_embedder, FAISSVectorStore

# Chunking
chunker = DocumentChunker(chunk_size=512, strategy="sentence")
chunks = chunker.chunk_text("Your document here")

# Embedding
embedder = create_embedder("sentence-transformer")
embeddings = embedder.embed([chunk.text for chunk in chunks])

# Storage & Retrieval
store = FAISSVectorStore(embedding_dim=384, metric="cosine")
store.add_batch(
    chunk_ids=[c.chunk_id for c in chunks],
    texts=[c.text for c in chunks],
    embeddings=embeddings
)

results = store.search(query_embedding, k=5)
```

### Guardrails

```python
from mazag_prototype.ai import GuardrailsSystem

guardrails = GuardrailsSystem(strict_mode=False)
result = guardrails.check_input("User message here")

if not result["allowed"]:
    if result["critical_issue"]:
        print(guardrails.get_crisis_response())
```

### Analysis

```python
from mazag_prototype.ai import create_analyzer

analyzer = create_analyzer("lexicon")
analysis = analyzer.analyze("I feel so sad and hopeless")

print(f"Sentiment: {analysis.sentiment}")
print(f"Emotions: {analysis.emotions}")
print(f"Risk Level: {analysis.risk_level}")
```

## Data Integration

### Using Vezeeta Psychiatrists Data

```python
import json

# Load therapist profiles
with open('vezeeta_psychiatrists.json', 'r', encoding='utf-8') as f:
    therapists = json.load(f)

# Get recommendations
recommendations = engine.get_therapist_recommendations(
    patient_concerns=["anxiety", "depression"],
    therapist_profiles=therapists,
    preferred_language="Arabic"
)
```

## Production Deployment (Next Steps)

1. **Scale Vector Store**: Use production database like FAISS

2. **Deploy as API**: FastAPI backend
   ```python
   from fastapi import FastAPI
   app = FastAPI()
   
   @app.post("/chat")
   async def chat(message: str):
       result = engine.process_message(message)
       return result
   ```

3. **Add Authentication**: User sessions and API keys

4. **Monitoring**: Track performance metrics
   - Response latency
   - Retrieval accuracy
   - Recommendation quality
   - Guardrail effectiveness

5. **Fine-tuning**: Custom embeddings for mental health domain

## Performance Benchmarks

### RAG Retrieval
- **Sentence Transformers**: ~50ms for embedding + retrieval
- **FAISS Search**: <1ms for 10K chunks
- **Chunk Processing**: ~100 chunks/second