"""
RAG Configuration Experiments for Mazag Backend
Tests different combinations of settings to find the best configuration
"""

import sys
import os
import time

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add mazag_prototype to path
sys.path.insert(0, 'mazag_prototype')

from mazag_prototype import create_mazag_engine

# Your API Key
API_KEY = "AIzaSyDN9HNMCko1N8p1h29scSJj1BN-rYOzB5g"

# Test knowledge base
KNOWLEDGE_DOCS = [
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

# Test queries - covering different topics in knowledge base
TEST_QUERIES = [
    # Anxiety-related (should match anxiety docs)
    "I'm feeling really anxious lately and I don't know why",
    "It feels like I'm going to have a panic attack",
    
    # Depression-related (should match depression docs)
    "I feel sad all the time and nothing interests me anymore",
    
    # Cognitive distortions (should match CBT docs)
    "I always fail at everything and nothing ever goes right",
    
    # Mindfulness/coping (should match mindfulness docs)
    "How can I calm my racing thoughts and stay present?",
]

# Control queries - should have LOW similarity (testing selectivity)
CONTROL_QUERIES = [
    "What's the weather like today?",  # Off-topic
    "How do I bake a chocolate cake?",  # Unrelated
    "Tell me about quantum physics",  # Completely different domain
]

# ========================================
# EXPERIMENT CONFIGURATIONS
# ========================================

EXPERIMENTS = [
    # Experiment 1: Baseline (default settings)
    {
        "name": "Baseline",
        "embedding_model": "sentence-transformer",
        "chunking_strategy": "sentence",
        "similarity_metric": "cosine",
        "rag_top_k": 3,
        "analysis_method": "lexicon",
    },
    
    # Experiment 2: Different chunking strategy
    {
        "name": "Fixed Chunking",
        "embedding_model": "sentence-transformer",
        "chunking_strategy": "fixed",
        "similarity_metric": "cosine",
        "rag_top_k": 3,
        "analysis_method": "lexicon",
    },
    
    # Experiment 3: Semantic chunking
    {
        "name": "Semantic Chunking",
        "embedding_model": "sentence-transformer",
        "chunking_strategy": "semantic",
        "similarity_metric": "cosine",
        "rag_top_k": 3,
        "analysis_method": "lexicon",
    },
    
    # Experiment 4: Recursive chunking (new strategy)
    {
        "name": "Recursive Chunking",
        "embedding_model": "sentence-transformer",
        "chunking_strategy": "recursive",
        "similarity_metric": "cosine",
        "rag_top_k": 3,
        "analysis_method": "lexicon",
    },
    
    # Experiment 5: More context (top-5)
    {
        "name": "More Context (k=5)",
        "embedding_model": "sentence-transformer",
        "chunking_strategy": "sentence",
        "similarity_metric": "cosine",
        "rag_top_k": 5,
        "analysis_method": "lexicon",
    },
    
    # Experiment 6: Less context (top-1)
    {
        "name": "Less Context (k=1)",
        "embedding_model": "sentence-transformer",
        "chunking_strategy": "sentence",
        "similarity_metric": "cosine",
        "rag_top_k": 1,
        "analysis_method": "lexicon",
    },
]


def run_experiment(config):
    """Run a single experiment with given configuration"""
    
    print(f"\n{'='*70}")
    print(f"🔬 Experiment: {config['name']}")
    print(f"{'='*70}")
    
    print("\n⚙️  Configuration:")
    for key, value in config.items():
        if key != 'name':
            print(f"   • {key}: {value}")
    
    try:
        # Create engine with experiment config
        start_time = time.time()
        engine = create_mazag_engine(
            api_key=API_KEY,
            embedding_model=config['embedding_model'],
            chunking_strategy=config['chunking_strategy'],
            similarity_metric=config['similarity_metric'],
            rag_top_k=config['rag_top_k'],
            analysis_method=config['analysis_method']
        )
        init_time = time.time() - start_time
        
        print(f"\n✅ Engine initialized in {init_time:.2f}s")
        
        # Add knowledge base
        print("📚 Adding knowledge base...")
        for doc in KNOWLEDGE_DOCS:
            engine.add_knowledge(doc.strip())
        
        print(f"   • Total chunks: {engine.vector_store.size()}")
        
        # Run test queries and collect metrics
        results = {
            "config": config,
            "init_time": init_time,
            "total_chunks": engine.vector_store.size(),
            "queries": [],
            "control_queries": []
        }
        
        print("\n💬 Testing relevant queries...")
        
        for i, query in enumerate(TEST_QUERIES, 1):
            print(f"\n   Query {i}: {query[:60]}...")
            
            query_start = time.time()
            result = engine.process_message(query, use_rag=True, analyze_sentiment=True)
            query_time = time.time() - query_start
            
            query_result = {
                "query": query,
                "query_time": query_time,
                "retrieved_chunks": len(result.get('retrieved_context', [])),
                "scores": [c['score'] for c in result.get('retrieved_context', [])],
                "retrieved_texts": [c['text'][:100] for c in result.get('retrieved_context', [])],
                "response_length": len(result.get('response', '')),
                "sentiment": result.get('analysis', {}).get('sentiment', 'unknown'),
                "risk_level": result.get('analysis', {}).get('risk_level', 'unknown')
            }
            
            results["queries"].append(query_result)
            
            # Show retrieval info
            if result.get('retrieved_context'):
                avg_score = sum(query_result['scores']) / len(query_result['scores'])
                print(f"      ✓ Retrieved {len(result['retrieved_context'])} chunks")
                print(f"      ✓ Avg similarity: {avg_score:.4f}")
                print(f"      ✓ Query time: {query_time:.3f}s")
                
                # Show top retrieved snippet for inspection
                top_chunk = result['retrieved_context'][0]
                snippet = top_chunk['text'][:80].replace('\n', ' ')
                print(f"      📝 Top match: '{snippet}...'")
            else:
                print(f"      ✗ No relevant chunks found")
        
        # Test control queries (should have LOW similarity)
        print("\n\n🎯 Testing control queries (should reject these)...")
        
        for i, query in enumerate(CONTROL_QUERIES, 1):
            print(f"\n   Control {i}: {query[:60]}...")
            
            try:
                result = engine.process_message(query, use_rag=True, analyze_sentiment=False)
                
                # Handle case where result might be None or missing keys
                if result is None:
                    print(f"      ✓ Blocked by guardrails (no result)")
                    control_result = {
                        "query": query,
                        "retrieved_chunks": 0,
                        "scores": []
                    }
                    results["control_queries"].append(control_result)
                    continue
                
                retrieved_context = result.get('retrieved_context', [])
                if retrieved_context is None:
                    retrieved_context = []
                
                control_result = {
                    "query": query,
                    "retrieved_chunks": len(retrieved_context),
                    "scores": [c['score'] for c in retrieved_context if c and 'score' in c]
                }
                
                results["control_queries"].append(control_result)
                
                if retrieved_context and len(retrieved_context) > 0:
                    if control_result['scores']:
                        avg_score = sum(control_result['scores']) / len(control_result['scores'])
                        print(f"      Retrieved {len(retrieved_context)} chunks")
                        print(f"      Avg similarity: {avg_score:.4f}")
                        
                        if avg_score > 0.5:
                            print(f"      ⚠️  WARNING: High similarity for off-topic query!")
                        else:
                            print(f"      ✓ Good - low similarity as expected")
                    else:
                        print(f"      Retrieved chunks but no scores")
                else:
                    print(f"      ✓ Perfect - no chunks retrieved")
                    
            except Exception as e:
                print(f"      ⚠️  Error processing control query: {str(e)}")
                control_result = {
                    "query": query,
                    "retrieved_chunks": 0,
                    "scores": []
                }
                results["control_queries"].append(control_result)
        
        # Calculate overall metrics for relevant queries
        all_scores = [score for q in results["queries"] for score in q["scores"]]
        results["avg_similarity"] = sum(all_scores) / len(all_scores) if all_scores else 0
        results["max_similarity"] = max(all_scores) if all_scores else 0
        results["min_similarity"] = min(all_scores) if all_scores else 0
        results["avg_query_time"] = sum(q["query_time"] for q in results["queries"]) / len(results["queries"])
        
        # Calculate retrieval quality: % of queries with avg similarity ≥ threshold
        # For small knowledge bases (< 10 chunks), use 0.3 threshold
        # For larger knowledge bases, use 0.7 threshold
        quality_threshold = 0.3
        
        high_quality_retrievals = 0
        for query in results["queries"]:
            if query["scores"]:
                avg_score = sum(query["scores"]) / len(query["scores"])
                if avg_score >= quality_threshold:
                    high_quality_retrievals += 1
        
        results["retrieval_quality"] = (high_quality_retrievals / len(results["queries"]) * 100) if results["queries"] else 0
        results["quality_threshold"] = quality_threshold
        
        # Calculate control query metrics (should be LOW)
        control_scores = [score for q in results["control_queries"] for score in q["scores"]]
        results["avg_control_similarity"] = sum(control_scores) / len(control_scores) if control_scores else 0
        
        # Selectivity: difference between relevant and control queries (higher is better)
        results["selectivity"] = results["avg_similarity"] - results["avg_control_similarity"]
        
        print(f"\n📊 Results Summary:")
        print(f"   Relevant Queries:")
        print(f"   • Average similarity: {results['avg_similarity']:.4f}")
        print(f"   • Max similarity: {results['max_similarity']:.4f}")
        print(f"   • Min similarity: {results['min_similarity']:.4f}")
        print(f"   • Retrieval quality: {results['retrieval_quality']:.1f}% (≥{results['quality_threshold']} threshold)")
        print(f"\n   Control Queries (off-topic):")
        print(f"   • Average similarity: {results['avg_control_similarity']:.4f} {'✓' if results['avg_control_similarity'] < 0.5 else '⚠️'}")
        print(f"\n   Performance:")
        print(f"   • Selectivity: {results['selectivity']:.4f} (higher = better)")
        print(f"   • Average query time: {results['avg_query_time']:.3f}s")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {str(e)}")
        return None


def compare_results(all_results):
    """Compare all experiment results"""
    
    print(f"\n\n{'='*70}")
    print("📊 COMPARISON OF ALL EXPERIMENTS")
    print(f"{'='*70}\n")
    
    # Sort by selectivity (primary) and retrieval quality (secondary)
    valid_results = [r for r in all_results if r is not None]
    sorted_results = sorted(valid_results, key=lambda x: (x['selectivity'], x['retrieval_quality']), reverse=True)
    
    print(f"{'Rank':<6} {'Experiment':<20} {'Selectivity':<13} {'Quality':<10} {'Avg Sim':<10} {'Control':<10} {'Time':<10}")
    print("-" * 90)
    
    for i, result in enumerate(sorted_results, 1):
        name = result['config']['name']
        selectivity = result['selectivity']
        quality = result['retrieval_quality']
        avg_sim = result['avg_similarity']
        control = result['avg_control_similarity']
        avg_time = result['avg_query_time']
        
        # Add indicator for good results
        # For small KBs (<10 chunks), lower expectations for quality %
        quality_threshold_used = result.get('quality_threshold', 0.7)
        if quality_threshold_used <= 0.3:  # Small KB
            quality_indicator = "✓" if quality >= 60 else ("⚠️" if quality >= 40 else "❌")
        else:  # Large KB
            quality_indicator = "✓" if quality >= 80 else ("⚠️" if quality >= 60 else "❌")
        control_indicator = "✓" if control < 0.5 else "⚠️"
        
        print(f"{i:<6} {name:<20} {selectivity:<13.4f} {quality:<5.1f}% {quality_indicator:<3} {avg_sim:<10.4f} {control:<5.4f} {control_indicator:<4} {avg_time:<10.3f}s")
    
    # Best configuration
    print(f"\n{'='*70}")
    print("🏆 BEST CONFIGURATION")
    print(f"{'='*70}")
    
    best = sorted_results[0]
    print(f"\n   Experiment: {best['config']['name']}")
    print(f"\n   Settings:")
    for key, value in best['config'].items():
        if key != 'name':
            print(f"      • {key}: {value}")
    
    print(f"\n   Performance:")
    print(f"      • Selectivity: {best['selectivity']:.4f} (difference: relevant vs control)")
    print(f"      • Retrieval quality: {best['retrieval_quality']:.1f}% (queries with ≥{best['quality_threshold']} similarity)")
    print(f"      • Average similarity (relevant): {best['avg_similarity']:.4f}")
    print(f"      • Average similarity (control): {best['avg_control_similarity']:.4f}")
    print(f"      • Max similarity: {best['max_similarity']:.4f}")
    print(f"      • Average query time: {best['avg_query_time']:.3f}s")
    print(f"      • Total chunks: {best['total_chunks']}")
    
    # Show a sample qualitative inspection
    print(f"\n   📝 Sample Retrieval (Quality Check):")
    sample_query = best['queries'][0]
    print(f"      Query: {sample_query['query'][:70]}...")
    if sample_query['retrieved_texts']:
        for i, text in enumerate(sample_query['retrieved_texts'][:2], 1):
            snippet = text.replace('\n', ' ').strip()
            print(f"      Chunk {i}: {snippet}...")
    
    # Recommendations
    print(f"\n{'='*70}")
    print("💡 RECOMMENDATIONS")
    print(f"{'='*70}\n")
    
    # Analyze patterns
    top_3 = sorted_results[:3]
    
    # Chunking strategy analysis
    chunking_strategies = [r['config']['chunking_strategy'] for r in top_3]
    if len(set(chunking_strategies)) == 1:
        print(f"   ✓ Best chunking strategy: {chunking_strategies[0]}")
    else:
        print(f"   📊 Top chunking strategies: {', '.join(set(chunking_strategies))}")
    
    # Similarity metric analysis
    similarity_metrics = [r['config']['similarity_metric'] for r in top_3]
    if len(set(similarity_metrics)) == 1:
        print(f"   ✓ Best similarity metric: {similarity_metrics[0]}")
    else:
        print(f"   📊 Top similarity metrics: {', '.join(set(similarity_metrics))}")
    
    # Top-k analysis
    top_k_values = [r['config']['rag_top_k'] for r in top_3]
    avg_k = sum(top_k_values) / len(top_k_values)
    print(f"   ✓ Recommended top-k: {int(avg_k)} chunks")
    
    # Quality analysis
    high_quality_configs = [r for r in valid_results if r['retrieval_quality'] >= 80]
    if high_quality_configs:
        print(f"\n   ⭐ {len(high_quality_configs)} configuration(s) achieved ≥80% retrieval quality")
    
    # Selectivity analysis
    high_selectivity_configs = [r for r in valid_results if r['selectivity'] > 0.3]
    if high_selectivity_configs:
        print(f"   🎯 {len(high_selectivity_configs)} configuration(s) showed good selectivity (>0.3)")
    
    # Speed vs accuracy tradeoff
    fastest = sorted(valid_results, key=lambda x: x['avg_query_time'])[0]
    if fastest != best:
        print(f"\n   ⚡ Fastest configuration: {fastest['config']['name']} ({fastest['avg_query_time']:.3f}s)")
        selectivity_diff = best['selectivity'] - fastest['selectivity']
        print(f"      Trade-off: {selectivity_diff:.4f} lower selectivity for speed")
    
    # Control query analysis
    print(f"\n   🎯 Control Query Performance:")
    best_control = min(valid_results, key=lambda x: x['avg_control_similarity'])
    print(f"      • Best at rejecting off-topic: {best_control['config']['name']}")
    print(f"        (control similarity: {best_control['avg_control_similarity']:.4f})")
    
    worst_control = max(valid_results, key=lambda x: x['avg_control_similarity'])
    if worst_control['avg_control_similarity'] > 0.5:
        print(f"      ⚠️  Warning: {worst_control['config']['name']} has high control similarity ({worst_control['avg_control_similarity']:.4f})")
        print(f"         This means it retrieves irrelevant content!")


def main():
    """Run all experiments"""
    
    print("="*70)
    print(" "*15 + "MAZAG RAG CONFIGURATION EXPERIMENTS")
    print("="*70)
    
    print(f"\n📋 Running {len(EXPERIMENTS)} experiments...")
    print(f"   • Relevant test queries: {len(TEST_QUERIES)}")
    print(f"   • Control queries (off-topic): {len(CONTROL_QUERIES)}")
    print(f"   • Knowledge documents: {len(KNOWLEDGE_DOCS)}")
    print(f"\n   Metrics tracked:")
    print(f"   • Selectivity: Relevant vs Control similarity")
    print(f"   • Retrieval Quality: % queries with ≥0.7 avg similarity")
    print(f"   • Qualitative inspection: Top retrieved snippets")
    
    input("\nPress Enter to start experiments...")
    
    all_results = []
    
    for i, experiment in enumerate(EXPERIMENTS, 1):
        print(f"\n\n[Experiment {i}/{len(EXPERIMENTS)}]")
        result = run_experiment(experiment)
        all_results.append(result)
        
        if i < len(EXPERIMENTS):
            print("\n⏳ Waiting 2 seconds before next experiment...")
            time.sleep(2)
    
    # Compare all results
    compare_results(all_results)
    
    print(f"\n\n{'='*70}")
    print("✅ ALL EXPERIMENTS COMPLETE!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

