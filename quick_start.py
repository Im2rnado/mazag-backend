"""
Quick Start Example for Mazag Backend
Demonstrates basic usage of the complete system
"""

import sys
import os

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add mazag_prototype to path
sys.path.insert(0, 'mazag_prototype')

from mazag_prototype import create_mazag_engine

def main():
    # Replace with your actual API key
    API_KEY = "AIzaSyDN9HNMCko1N8p1h29scSJj1BN-rYOzB5g"
    
    print("🚀 Initializing Mazag Engine...\n")
    
    # Create engine with default settings
    print("⚙️  Configuration:")
    print("   • Embedding Model: sentence-transformer (all-MiniLM-L6-v2)")
    print("   • Analysis Method: lexicon-based")
    print("   • Chunking Strategy: sentence-based")
    print("   • Similarity Metric: cosine")
    print("   • RAG Top-K: 1 chunks\n")
    
    engine = create_mazag_engine(
        api_key=API_KEY,
        embedding_model="sentence-transformer",
        chunking_strategy="sentence",
        similarity_metric="cosine",
        rag_top_k=1,
        analysis_method="lexicon"
    )
    
    print("✅ Engine initialized!\n")
    
    # Add some mental health knowledge
    print("📚 Adding knowledge base...")
    
    knowledge_docs = [
        """
        Anxiety disorders are characterized by excessive worry and fear.
        Common symptoms include restlessness, rapid heartbeat, and difficulty concentrating.
        Treatment often involves cognitive behavioral therapy and relaxation techniques.
        """,
        """
        Depression symptoms include persistent sadness, loss of interest in activities,
        changes in appetite, sleep disturbances, and difficulty concentrating.
        It's important to seek professional help if symptoms persist.
        """,
        """
        Cognitive distortions are irrational thought patterns that can worsen mental health.
        Examples include all-or-nothing thinking, catastrophizing, and overgeneralization.
        CBT helps identify and challenge these distortions.
        """,
        """
        Mindfulness meditation can reduce stress and anxiety by helping you stay present
        in the moment. It involves observing thoughts and feelings without judgment.
        Regular practice can improve emotional regulation.
        """,
        """
        Panic attacks are sudden episodes of intense fear that trigger severe physical
        reactions such as rapid heartbeat, sweating, trembling, and shortness of breath.
        Learning breathing exercises and grounding techniques can help manage them.
        """,
        """
        Stress management involves identifying sources of stress and developing healthy
        coping mechanisms. Regular exercise, balanced nutrition, and adequate sleep play
        a major role in maintaining mental well-being.
        """,
        """
        Social anxiety disorder involves an intense fear of being judged or embarrassed
        in social situations. Gradual exposure, social skills training, and CBT are
        effective treatment approaches.
        """,
        """
        Sleep hygiene refers to practices that improve sleep quality, such as maintaining
        a regular bedtime, avoiding screens before sleep, and limiting caffeine intake.
        Poor sleep can worsen anxiety and depression.
        """,
        """
        Grounding techniques help individuals manage overwhelming emotions by focusing
        on the present moment. Examples include deep breathing, describing surroundings,
        or holding a cold object to redirect attention.
        """,
        """
        Self-care is the practice of taking time to care for your physical and emotional
        needs. It includes activities like journaling, spending time outdoors, connecting
        with supportive people, and setting healthy boundaries.
        """,
        """
        Emotional regulation involves recognizing, understanding, and managing emotions.
        Strategies include mindfulness, reframing negative thoughts, and problem-solving.
        Improved regulation reduces impulsive reactions and emotional distress.
        """,
        """
        If someone expresses suicidal thoughts, it's important to take them seriously.
        Encourage them to reach out for help, and never leave them alone if they are in danger.
        In Egypt, contact the suicide hotline on 16328.
        """
    ]
    
    print(f"   Processing {len(knowledge_docs)} documents...\n")
    
    for i, doc in enumerate(knowledge_docs, 1):
        doc_text = doc.strip()
        print(f"   Document {i}: {len(doc_text)} characters")
        
        # Get chunks before adding
        chunks = engine.chunker.chunk_text(doc_text)
        print(f"   └─ Split into {len(chunks)} chunks")
        
        engine.add_knowledge(doc_text)
    
    print(f"\n✅ Knowledge base ready!")
    print(f"   • Total chunks in vector store: {engine.vector_store.size()}")
    print(f"   • Embedding dimension: {engine.embedder.get_embedding_dim()}")
    print(f"   • Ready for semantic search\n")
    
    # Example conversation
    print("💬 Example Conversation:\n")
    print("=" * 60)
    
    test_messages = [
        "I'm feeling really anxious lately and I don't know why",
        "It feels like I'm going to have a panic attack",
        "I'm having trouble sleeping too"
    ]
    
    # Track RAG statistics
    rag_stats = {
        "total_retrievals": 0,
        "total_chunks_retrieved": 0,
        "avg_scores": []
    }
    
    for message in test_messages:
        print(f"\n👤 User: {message}")
        print("-" * 60)
        
        # Process message through complete pipeline
        result = engine.process_message(
            message,
            use_rag=True,
            analyze_sentiment=True
        )
        
        # Show RAG retrieval details
        if result['retrieved_context']:
            print(f"\n🔍 RAG Retrieval:")
            print(f"   Retrieved {len(result['retrieved_context'])} chunks from knowledge base")
            
            # Track stats
            rag_stats["total_retrievals"] += 1
            rag_stats["total_chunks_retrieved"] += len(result['retrieved_context'])
            
            for i, chunk in enumerate(result['retrieved_context'], 1):
                score = chunk['score']
                rag_stats["avg_scores"].append(score)
                
                print(f"\n   Chunk {i} (Score: {score:.4f}):")
                # Show first 100 chars of the chunk
                chunk_preview = chunk['text'].replace('\n', ' ').strip()[:100]
                print(f"   └─ {chunk_preview}...")
            
            # Show total context length
            total_context = "\n\n".join([c['text'] for c in result['retrieved_context']])
            print(f"\n   📝 Total context length: {len(total_context)} characters")
        else:
            print(f"\n🔍 RAG Retrieval: No relevant chunks found")
        
        # Show analysis
        if result['analysis']:
            analysis = result['analysis']
            print(f"\n📊 Analysis:")
            print(f"   • Sentiment: {analysis['sentiment']}")
            print(f"   • Tone: {analysis['tone']}")
            print(f"   • Risk Level: {analysis['risk_level']}")
            if analysis['indicators']:
                print(f"   • Indicators: {', '.join(analysis['indicators'])}")
        
        # Show response
        print(f"\n🤖 Mazag: {result['response']}")
        print()
    
    print("=" * 60)
    print("\n✅ Demo complete!")
    
    # Show RAG performance summary
    if rag_stats["avg_scores"]:
        print("\n📊 RAG Performance Summary:")
        print(f"   • Total queries: {len(test_messages)}")
        print(f"   • Successful retrievals: {rag_stats['total_retrievals']}")
        print(f"   • Total chunks retrieved: {rag_stats['total_chunks_retrieved']}")
        print(f"   • Average chunks per query: {rag_stats['total_chunks_retrieved'] / rag_stats['total_retrievals']:.1f}")
        print(f"   • Average similarity score: {sum(rag_stats['avg_scores']) / len(rag_stats['avg_scores']):.4f}")
        print(f"   • Highest similarity score: {max(rag_stats['avg_scores']):.4f}")
        print(f"   • Lowest similarity score: {min(rag_stats['avg_scores']):.4f}")


if __name__ == "__main__":
    main()

