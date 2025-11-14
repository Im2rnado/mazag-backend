"""
Component Testing Script for Mazag Backend
Tests each component independently
"""

import sys

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.insert(0, 'mazag_prototype')

def test_chunking():
    """Test document chunking"""
    print("\n" + "="*60)
    print("Testing: Document Chunking")
    print("="*60)
    
    from mazag_prototype.rag import DocumentChunker
    
    sample_text = """
    Cognitive Behavioral Therapy (CBT) is an effective treatment for anxiety and depression.
    It helps identify negative thought patterns and replace them with healthier ones.

    Common CBT techniques include thought records, behavioral activation, and exposure therapy.
    These methods have been proven effective through extensive research.

    CBT sessions are typically structured, goal-oriented, and involve collaboration between therapist and client.
    Clients often receive homework assignments to practice skills learned during therapy.

    In addition to CBT, mindfulness-based interventions have gained popularity for managing stress and emotional regulation.
    Mindfulness encourages individuals to observe their thoughts and feelings without judgment, fostering greater self-awareness.

    Techniques such as cognitive restructuring, relaxation training, and systematic desensitization are also employed to help individuals manage anxiety.
ch
    Studies have shown that integrating CBT with medication can further improve outcomes, especially for those with moderate to severe symptoms.
    """
    
    chunker = DocumentChunker(chunk_size=100, overlap=4, strategy="semantic")
    chunks = chunker.chunk_text(sample_text.strip())
    
    print(f"\n🔍 Sample text: {sample_text}")
    print(f"\n✅ Created {len(chunks)} chunks")
    print(f"\nFirst chunk preview:")
    print(f"   ID: {chunks[0].chunk_id}")
    print(f"   Text: {chunks[0].text}")
    print(f"   Length: {len(chunks[0].text)} chars")
    print(f"\nSecond chunk preview:")
    print(f"   ID: {chunks[1].chunk_id}")
    print(f"   Text: {chunks[1].text}")
    print(f"   Length: {len(chunks[1].text)} chars")

def test_embedding():
    """Test text embedding"""
    print("\n" + "="*60)
    print("Testing: Text Embedding")
    print("="*60)
    
    from mazag_prototype.rag import create_embedder
    
    embedder = create_embedder("sentence-transformer", model_name="all-MiniLM-L6-v2")
    
    test_texts = [
        "I feel anxious",
        "I am depressed",
        "I need help"
    ]
    
    embeddings = embedder.embed(test_texts)
    
    print(f"\n✅ Generated embeddings")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Dimension: {embedder.get_embedding_dim()}")
    
    for text, embedding in zip(test_texts, embeddings):
        print(f"   {text}: [{embedding[:5]}...]")


def test_vector_store():
    """Test vector storage and retrieval"""
    print("\n" + "="*60)
    print("Testing: Vector Store & Retrieval")
    print("="*60)
    
    from mazag_prototype.rag import create_embedder, SimpleVectorStore
    
    embedder = create_embedder("sentence-transformer")
    store = SimpleVectorStore(metric="cosine")
    
    # Add some documents
    docs = [
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
    
    embeddings = embedder.embed(docs)
    
    for i, (doc, emb) in enumerate(zip(docs, embeddings)):
        store.add(f"doc_{i}", doc, emb)
    
    print(f"\n✅ Added {store.size()} documents")
    print(f"   First document: {store.chunks[0].text}")
    print(f"   First embedding: {store.chunks[0].embedding[:5]}...")
    print(f"   Second document: {store.chunks[1].text}")
    print(f"   Second embedding: {store.chunks[1].embedding[:5]}...")
    print(f"   Third document: {store.chunks[2].text}")
    print(f"   Third embedding: {store.chunks[2].embedding[:5]}...")
    
    # Test retrieval
    query = "I feel like I worry a lot"
    query_emb = embedder.embed(query)
    results = store.search(query_emb, k=3)
    
    print(f"\n🔍 Query: '{query}'")
    print(f"\nResults:")
    for i, result in enumerate(results, 1):
        print(f"   {i}. Score: {result['score']:.3f} | {result['text']}")


def test_guardrails():
    """Test guardrails system"""
    print("\n" + "="*60)
    print("Testing: Guardrails System")
    print("="*60)
    
    from mazag_prototype.ai import GuardrailsSystem
    
    guardrails = GuardrailsSystem()
    
    test_cases = [
        ("I'm feeling anxious", True),  # Should pass
        ("Buy crypto now!", False),      # Should block (off-topic)
        ("I want to die", False),        # Should block (critical)
    ]
    
    for text, should_pass in test_cases:
        result = guardrails.check_input(text)
        status = "✅ PASS" if result['allowed'] else "❌ BLOCK"
        expected = "✅" if (result['allowed'] == should_pass) else "⚠️ UNEXPECTED"
        
        print(f"\n{expected} Input: '{text}'")
        print(f"   Status: {status}")
        if not result['allowed']:
            print(f"   Reason: {result['results'][0]['reason'][:50]}...")


def test_analyzer():
    """Test sentiment analyzer"""
    print("\n" + "="*60)
    print("Testing: Sentiment Analysis")
    print("="*60)
    
    from mazag_prototype.ai import create_analyzer
    
    analyzer = create_analyzer("lexicon")
    
    test_texts = [
        "I'm so happy and grateful!",
        "I feel sad and hopeless",
        "I'm worried about everything"
    ]
    
    for text in test_texts:
        analysis = analyzer.analyze(text)
        print(f"\n💬 '{text}'")
        print(f"   Sentiment: {analysis.sentiment}")
        print(f"   Tone: {analysis.tone}")
        print(f"   Risk: {analysis.risk_level}")


def test_recommender():
    """Test recommendation system"""
    print("\n" + "="*60)
    print("Testing: Therapist Recommendation")
    print("="*60)
    
    from mazag_prototype.recommend import recommend_therapists
    
    # Sample therapist data
    therapists = [
        {
            "id": "t1",
            "name": "Dr. Ahmed Hassan",
            "specialties": ["Anxiety", "Depression", "CBT"],
            "description": "Specializes in anxiety and depression using CBT",
            "languages": ["Arabic", "English"],
            "approach": ["CBT"]
        },
        {
            "id": "t2",
            "name": "Dr. Sara Mohamed",
            "specialties": ["Family Counseling", "Relationship Issues"],
            "description": "Expert in family therapy and relationships",
            "languages": ["Arabic"],
            "approach": ["Family Therapy"]
        },
        {
            "id": "t3",
            "name": "Dr. John Smith",
            "specialties": ["Anxiety", "Panic", "Mindfulness"],
            "description": "Treats anxiety and panic with mindfulness techniques",
            "languages": ["English"],
            "approach": ["Mindfulness"]
        }
    ]
    
    patient_desc = "I have severe anxiety and panic attacks"
    
    recommendations = recommend_therapists(
        patient_description=patient_desc,
        therapist_profiles=therapists,
        top_k=3
    )
    
    print(f"\n👤 Patient: '{patient_desc}'")
    print(f"\n🎯 Top Recommendations:")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n   {i}. {rec['therapist']['name']}")
        print(f"      Score: {rec['score']:.2f}")
        print(f"      Reasons: {', '.join(rec['reasons'])}")


def main():
    """Run all component tests"""
    print("\n" + "="*70)
    print(" "*20 + "MAZAG COMPONENT TESTS")
    print("="*70)
    
    tests = [
        ("Chunking", test_chunking),
        # ("Embedding", test_embedding),
        # ("Vector Store", test_vector_store),
        # ("Guardrails", test_guardrails),
        # ("Analyzer", test_analyzer),
        # ("Recommender", test_recommender)
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ {name} test FAILED: {str(e)}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"\nTest Results: {passed} passed, {failed} failed")
    print("\n" + "="*70)
    

if __name__ == "__main__":
    main()

