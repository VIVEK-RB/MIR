"""
Multilingual Audio Retrieval System
Supports English (LibriSpeech) and Tamil (Indic TTS)
Uses CLAP embeddings for both audio and text
"""

import numpy as np
import json
import faiss
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MultilingualAudioRetrieval:
    """
    Unified retrieval system for English and Tamil audio
    """
    
    def __init__(self):
        self.english_data = {}
        self.tamil_data = {}
        self.combined_audio_index = None
        self.combined_text_index = None
        self.combined_metadata = []
        
    def load_english_data(self, 
                         audio_emb_path: str = "preprocessed_features/audio_features/clap_embeddings.npy",
                         text_emb_path: str = "preprocessed_features/text_features/clap_text_embeddings.npy",
                         metadata_path: str = "preprocessed_features/metadata/dataset_metadata.json"):
        """Load English (LibriSpeech) embeddings and metadata"""
        
        print("\n" + "="*70)
        print("Loading English (LibriSpeech) Data")
        print("="*70)
        
        try:
            # Load embeddings
            self.english_data['audio_embeddings'] = np.load(audio_emb_path)
            self.english_data['text_embeddings'] = np.load(text_emb_path)
            
            # Load metadata
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.english_data['metadata'] = json.load(f)
            
            # Add language tag
            for item in self.english_data['metadata']:
                item['language'] = 'english'
            
            print(f"✓ Audio embeddings: {self.english_data['audio_embeddings'].shape}")
            print(f"✓ Text embeddings: {self.english_data['text_embeddings'].shape}")
            print(f"✓ Metadata entries: {len(self.english_data['metadata'])}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading English data: {e}")
            return False
    
    def load_tamil_data(self,
                       audio_emb_path: str = "tamil_features/CLAP_audio_embeddings.npy",
                       text_emb_path: str = "tamil_features/clap_text_embeddings.npy",
                       metadata_path: str = "tamil_features/metadata.json"):
        """Load Tamil (Indic TTS) embeddings and metadata"""
        
        print("\n" + "="*70)
        print("Loading Tamil (Indic TTS) Data")
        print("="*70)
        
        try:
            # Load embeddings
            self.tamil_data['audio_embeddings'] = np.load(audio_emb_path)
            self.tamil_data['text_embeddings'] = np.load(text_emb_path)
            
            # Load metadata
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.tamil_data['metadata'] = json.load(f)
            
            # Add language tag
            for item in self.tamil_data['metadata']:
                item['language'] = 'tamil'
            
            print(f"✓ Audio embeddings: {self.tamil_data['audio_embeddings'].shape}")
            print(f"✓ Text embeddings: {self.tamil_data['text_embeddings'].shape}")
            print(f"✓ Metadata entries: {len(self.tamil_data['metadata'])}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading Tamil data: {e}")
            return False
    
    def build_unified_index(self):
        """Build unified FAISS indices for both languages"""
        
        print("\n" + "="*70)
        print("Building Unified Multilingual Index")
        print("="*70)
        
        # Combine audio embeddings
        audio_embeddings_list = []
        text_embeddings_list = []
        metadata_list = []
        
        if self.english_data:
            audio_embeddings_list.append(self.english_data['audio_embeddings'])
            text_embeddings_list.append(self.english_data['text_embeddings'])
            metadata_list.extend(self.english_data['metadata'])
        
        if self.tamil_data:
            audio_embeddings_list.append(self.tamil_data['audio_embeddings'])
            text_embeddings_list.append(self.tamil_data['text_embeddings'])
            metadata_list.extend(self.tamil_data['metadata'])
        
        # Combine embeddings
        combined_audio_emb = np.vstack(audio_embeddings_list).astype('float32')
        combined_text_emb = np.vstack(text_embeddings_list).astype('float32')
        
        print(f"\nCombined audio embeddings: {combined_audio_emb.shape}")
        print(f"Combined text embeddings: {combined_text_emb.shape}")
        print(f"Total entries: {len(metadata_list)}")
        
        # Normalize embeddings (important for cosine similarity)
        faiss.normalize_L2(combined_audio_emb)
        faiss.normalize_L2(combined_text_emb)
        
        # Build FAISS indices (using Inner Product for normalized vectors = cosine similarity)
        embedding_dim_audio = combined_audio_emb.shape[1]
        embedding_dim_text = combined_text_emb.shape[1]
        
        self.combined_audio_index = faiss.IndexFlatIP(embedding_dim_audio)
        self.combined_audio_index.add(combined_audio_emb)
        
        self.combined_text_index = faiss.IndexFlatIP(embedding_dim_text)
        self.combined_text_index.add(combined_text_emb)
        
        self.combined_metadata = metadata_list
        
        print(f"\n✓ Audio index built: {self.combined_audio_index.ntotal} vectors")
        print(f"✓ Text index built: {self.combined_text_index.ntotal} vectors")
        
        # Distribution by language
        english_count = sum(1 for m in metadata_list if m['language'] == 'english')
        tamil_count = sum(1 for m in metadata_list if m['language'] == 'tamil')
        
        print(f"\nLanguage distribution:")
        print(f"  English: {english_count} ({english_count/len(metadata_list)*100:.1f}%)")
        print(f"  Tamil: {tamil_count} ({tamil_count/len(metadata_list)*100:.1f}%)")
    
    def search_by_text(self, query: str, top_k: int = 10, 
                       language_filter: Optional[str] = None) -> List[Dict]:
        """
        Search audio by text query (English or Tamil)
        
        Args:
            query: Text query (can be English or Tamil)
            top_k: Number of results
            language_filter: Filter by 'english', 'tamil', or None (search both)
            
        Returns:
            List of results with scores
        """
        
        print(f"\nSearching by text: '{query[:50]}...'")
        print(f"Language filter: {language_filter if language_filter else 'All languages'}")
        
        # Since we don't have CLAP loaded for real-time encoding,
        # we'll use TF-IDF based text matching as fallback
        results = self._text_search_fallback(query, top_k, language_filter)
        
        return results
    
    def _text_search_fallback(self, query: str, top_k: int, 
                              language_filter: Optional[str] = None) -> List[Dict]:
        """
        Fallback text search using simple text matching
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Filter metadata by language if needed
        if language_filter:
            candidates = [m for m in self.combined_metadata if m['language'] == language_filter]
        else:
            candidates = self.combined_metadata
        
        if not candidates:
            return []
        
        # Extract texts
        texts = []
        for item in candidates:
            text = item.get('transcript') or item.get('text', '')
            texts.append(text.lower())
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(max_features=1000)
        
        try:
            # Fit on corpus + query
            all_texts = texts + [query.lower()]
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # Query is the last vector
            query_vec = tfidf_matrix[-1]
            corpus_vecs = tfidf_matrix[:-1]
            
            # Calculate similarities
            similarities = cosine_similarity(query_vec, corpus_vecs).flatten()
            
            # Get top k
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                result = candidates[idx].copy()
                result['similarity_score'] = float(similarities[idx])
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"⚠️  Text search error: {e}")
            # Return first k results as fallback
            return [dict(m, similarity_score=0.0) for m in candidates[:top_k]]
    
    def search_similar_audio(self, reference_idx: int, top_k: int = 10,
                            language_filter: Optional[str] = None) -> List[Dict]:
        """
        Find similar audio by providing a reference index
        
        Args:
            reference_idx: Index of reference audio
            top_k: Number of results
            language_filter: Filter by language
            
        Returns:
            List of similar audio results
        """
        
        if reference_idx >= len(self.combined_metadata):
            print(f"❌ Invalid reference index: {reference_idx}")
            return []
        
        ref_meta = self.combined_metadata[reference_idx]
        print(f"\nFinding audio similar to:")
        print(f"  Language: {ref_meta['language']}")
        print(f"  ID: {ref_meta.get('file_id') or ref_meta.get('id')}")
        
        # Get embedding for reference audio
        ref_emb = self.combined_audio_index.reconstruct(reference_idx).reshape(1, -1)
        
        # Search
        search_k = top_k * 5 if language_filter else top_k + 1  # +1 to exclude self
        scores, indices = self.combined_audio_index.search(ref_emb, search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            # Skip the reference itself
            if idx == reference_idx:
                continue
                
            if idx < len(self.combined_metadata):
                result = self.combined_metadata[idx].copy()
                result['similarity_score'] = float(score)
                
                # Apply language filter
                if language_filter is None or result['language'] == language_filter:
                    results.append(result)
                    
                if len(results) >= top_k:
                    break
        
        return results
    
    def cross_lingual_search(self, query_language: str, target_language: str, 
                            query_idx: int, top_k: int = 10) -> List[Dict]:
        """
        Cross-lingual search: Find Tamil audio from English query or vice versa
        
        Args:
            query_language: 'english' or 'tamil'
            target_language: 'english' or 'tamil'
            query_idx: Index of query audio in its language dataset
            top_k: Number of results
        """
        
        if query_language == target_language:
            print("⚠️  Use search_similar_audio for same-language search")
            return self.search_similar_audio(query_idx, top_k, target_language)
        
        # Find actual index in combined dataset
        actual_idx = query_idx
        if query_language == 'tamil' and self.english_data:
            actual_idx += len(self.english_data['metadata'])
        
        if actual_idx >= len(self.combined_metadata):
            print(f"❌ Invalid query index")
            return []
        
        print(f"\nCross-lingual search:")
        print(f"  Query: {query_language}")
        print(f"  Target: {target_language}")
        
        return self.search_similar_audio(actual_idx, top_k, target_language)
    
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        
        stats = {
            'total_entries': len(self.combined_metadata),
            'english_entries': sum(1 for m in self.combined_metadata if m['language'] == 'english'),
            'tamil_entries': sum(1 for m in self.combined_metadata if m['language'] == 'tamil'),
            'audio_embedding_dim': self.combined_audio_index.d if self.combined_audio_index else 0,
            'text_embedding_dim': self.combined_text_index.d if self.combined_text_index else 0,
        }
        
        return stats
    
    def save_index(self, output_path: str = "unified_index"):
        """Save unified index for later use"""
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving unified index to {output_path}/")
        
        # Save FAISS indices
        faiss.write_index(self.combined_audio_index, str(output_path / "audio_index.faiss"))
        faiss.write_index(self.combined_text_index, str(output_path / "text_index.faiss"))
        
        # Save metadata
        with open(output_path / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(self.combined_metadata, f, indent=2, ensure_ascii=False)
        
        # Save statistics
        stats = self.get_statistics()
        with open(output_path / "statistics.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        print("✓ Index saved successfully")
    
    def load_index(self, index_path: str = "unified_index"):
        """Load pre-built unified index"""
        
        index_path = Path(index_path)
        
        print(f"\nLoading unified index from {index_path}/")
        
        # Load FAISS indices
        self.combined_audio_index = faiss.read_index(str(index_path / "audio_index.faiss"))
        self.combined_text_index = faiss.read_index(str(index_path / "text_index.faiss"))
        
        # Load metadata
        with open(index_path / "metadata.json", 'r', encoding='utf-8') as f:
            self.combined_metadata = json.load(f)
        
        print("✓ Index loaded successfully")
        print(f"  Total entries: {len(self.combined_metadata)}")


def main():
    """Example usage"""
    
    print("\n" + "="*70)
    print("MULTILINGUAL AUDIO RETRIEVAL SYSTEM")
    print("English (LibriSpeech) + Tamil (Indic TTS)")
    print("="*70)
    
    # Initialize system
    retrieval = MultilingualAudioRetrieval()
    
    # Load English data
    english_loaded = retrieval.load_english_data(
        audio_emb_path="preprocessed_features/audio_features/clap_embeddings.npy",
        text_emb_path="preprocessed_features/text_features/clap_text_embeddings.npy",
        metadata_path="preprocessed_features/metadata/dataset_metadata.json"
    )
    
    # Load Tamil data
    tamil_loaded = retrieval.load_tamil_data(
        audio_emb_path="tamil_features/CLAP_audio_embeddings.npy",
        text_emb_path="tamil_features/clap_text_embeddings.npy",
        metadata_path="tamil_features/metadata.json"
    )
    
    if not (english_loaded or tamil_loaded):
        print("\n❌ Failed to load any data. Please check file paths.")
        return
    
    # Build unified index
    retrieval.build_unified_index()
    
    # Save index
    retrieval.save_index("unified_index")
    
    # Display statistics
    print("\n" + "="*70)
    print("SYSTEM STATISTICS")
    print("="*70)
    stats = retrieval.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Example searches
    print("\n" + "="*70)
    print("EXAMPLE SEARCHES")
    print("="*70)
    
    # 1. Search in English only
    print("\n1. Searching English audio only:")
    results = retrieval.search_by_text("speech", top_k=3, language_filter='english')
    for i, result in enumerate(results, 1):
        print(f"  [{i}] {result.get('file_id', result.get('id'))} - Score: {result['similarity_score']:.4f}")
        print(f"      Text: {result.get('transcript', result.get('text', ''))[:60]}...")
    
    # 2. Search in Tamil only
    print("\n2. Searching Tamil audio only:")
    results = retrieval.search_by_text("தமிழ்", top_k=3, language_filter='tamil')
    for i, result in enumerate(results, 1):
        print(f"  [{i}] {result.get('id')} - Score: {result['similarity_score']:.4f}")
        print(f"      Text: {result.get('text', '')[:60]}...")
    
    # 3. Find similar audio (English)
    if english_loaded:
        print("\n3. Finding similar English audio (reference index 0):")
        results = retrieval.search_similar_audio(0, top_k=3, language_filter='english')
        for i, result in enumerate(results, 1):
            print(f"  [{i}] {result.get('file_id')} - Score: {result['similarity_score']:.4f}")
    
    # 4. Cross-lingual search
    if english_loaded and tamil_loaded:
        print("\n4. Cross-lingual search (English → Tamil):")
        results = retrieval.cross_lingual_search('english', 'tamil', 0, top_k=3)
        for i, result in enumerate(results, 1):
            print(f"  [{i}] {result.get('id')} - Score: {result['similarity_score']:.4f}")
    
    print("\n" + "="*70)
    print("✅ SYSTEM READY FOR QUERIES")
    print("="*70)


if __name__ == "__main__":
    main()
