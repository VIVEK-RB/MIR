"""
Interactive Query Interface for Multilingual Audio Retrieval
"""

import sys
from pathlib import Path
import numpy as np
import json

# Import the retrieval system
# from multilingual_retrieval_system import MultilingualAudioRetrieval


class InteractiveRetrieval:
    """Interactive interface for querying the retrieval system"""
    
    def __init__(self, retrieval_system):
        self.system = retrieval_system
    
    def display_menu(self):
        """Display main menu"""
        print("\n" + "="*70)
        print("MULTILINGUAL AUDIO RETRIEVAL - INTERACTIVE MODE")
        print("="*70)
        print("\nOptions:")
        print("  1. Search by text (all languages)")
        print("  2. Search by text (English only)")
        print("  3. Search by text (Tamil only)")
        # print("  4. Find similar audio by index")
        # print("  5. Cross-lingual search (English → Tamil)")
        # print("  6. Cross-lingual search (Tamil → English)")
        # print("  7. View system statistics")
        # print("  8. Browse random samples")
        # print("  9. Export search results to CSV")
        # print("  0. Exit")
        print("="*70)
    
    def search_text_all(self):
        """Search by text across all languages"""
        query = input("\nEnter search query (English or Tamil): ").strip()
        if not query:
            print("❌ Empty query")
            return
        
        top_k = int(input("Number of results (default 10): ") or "10")
        
        results = self.system.search_by_text(query, top_k=top_k)
        self.display_results(results)
    
    def search_text_english(self):
        """Search English audio only"""
        query = input("\nEnter search query (English): ").strip()
        if not query:
            print("❌ Empty query")
            return
        
        top_k = int(input("Number of results (default 10): ") or "10")
        
        results = self.system.search_by_text(query, top_k=top_k, language_filter='english')
        self.display_results(results)
    
    def search_text_tamil(self):
        """Search Tamil audio only"""
        query = input("\nEnter search query (Tamil): ").strip()
        if not query:
            print("❌ Empty query")
            return
        
        top_k = int(input("Number of results (default 10): ") or "10")
        
        results = self.system.search_by_text(query, top_k=top_k, language_filter='tamil')
        self.display_results(results)
    
    def search_similar_audio(self):
        """Find similar audio by index"""
        print(f"\nTotal entries: {len(self.system.combined_metadata)}")
        
        idx = int(input("Enter reference audio index: "))
        if idx < 0 or idx >= len(self.system.combined_metadata):
            print(f"❌ Invalid index. Must be 0-{len(self.system.combined_metadata)-1}")
            return
        
        # Show reference audio info
        ref = self.system.combined_metadata[idx]
        print(f"\nReference audio:")
        print(f"  Language: {ref['language']}")
        print(f"  ID: {ref.get('file_id') or ref.get('id')}")
        print(f"  Text: {ref.get('transcript') or ref.get('text', '')[:80]}...")
        
        lang_filter = input("\nFilter by language? (english/tamil/all) [all]: ").strip().lower()
        if lang_filter not in ['english', 'tamil', 'all']:
            lang_filter = None
        elif lang_filter == 'all':
            lang_filter = None
        
        top_k = int(input("Number of results (default 10): ") or "10")
        
        results = self.system.search_similar_audio(idx, top_k=top_k, language_filter=lang_filter)
        self.display_results(results)
    
    def cross_lingual_en_to_ta(self):
        """Cross-lingual: English to Tamil"""
        if not self.system.english_data or not self.system.tamil_data:
            print("❌ Both English and Tamil data required")
            return
        
        max_idx = len(self.system.english_data['metadata']) - 1
        print(f"\nEnglish audio indices: 0-{max_idx}")
        
        idx = int(input("Enter English audio index: "))
        if idx < 0 or idx > max_idx:
            print(f"❌ Invalid index")
            return
        
        # Show query audio
        ref = self.system.english_data['metadata'][idx]
        print(f"\nQuery (English):")
        print(f"  ID: {ref['file_id']}")
        print(f"  Text: {ref.get('transcript', '')[:80]}...")
        
        top_k = int(input("Number of Tamil results (default 10): ") or "10")
        
        results = self.system.cross_lingual_search('english', 'tamil', idx, top_k=top_k)
        self.display_results(results, title="Similar Tamil Audio")
    
    def cross_lingual_ta_to_en(self):
        """Cross-lingual: Tamil to English"""
        if not self.system.english_data or not self.system.tamil_data:
            print("❌ Both English and Tamil data required")
            return
        
        max_idx = len(self.system.tamil_data['metadata']) - 1
        print(f"\nTamil audio indices: 0-{max_idx}")
        
        idx = int(input("Enter Tamil audio index: "))
        if idx < 0 or idx > max_idx:
            print(f"❌ Invalid index")
            return
        
        # Show query audio
        ref = self.system.tamil_data['metadata'][idx]
        print(f"\nQuery (Tamil):")
        print(f"  ID: {ref['id']}")
        print(f"  Text: {ref.get('text', '')[:80]}...")
        
        top_k = int(input("Number of English results (default 10): ") or "10")
        
        results = self.system.cross_lingual_search('tamil', 'english', idx, top_k=top_k)
        self.display_results(results, title="Similar English Audio")
    
    def view_statistics(self):
        """Display system statistics"""
        stats = self.system.get_statistics()
        
        print("\n" + "="*70)
        print("SYSTEM STATISTICS")
        print("="*70)
        
        for key, value in stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print("\n" + "="*70)
    
    def browse_samples(self):
        """Browse random samples"""
        import random
        
        num_samples = int(input("\nNumber of samples to show (default 5): ") or "5")
        lang = input("Language (english/tamil/all) [all]: ").strip().lower()
        
        # Filter by language
        if lang == 'english':
            candidates = [i for i, m in enumerate(self.system.combined_metadata) 
                         if m['language'] == 'english']
        elif lang == 'tamil':
            candidates = [i for i, m in enumerate(self.system.combined_metadata) 
                         if m['language'] == 'tamil']
        else:
            candidates = list(range(len(self.system.combined_metadata)))
        
        # Random sample
        sample_indices = random.sample(candidates, min(num_samples, len(candidates)))
        
        print("\n" + "="*70)
        print("RANDOM SAMPLES")
        print("="*70)
        
        for i, idx in enumerate(sample_indices, 1):
            item = self.system.combined_metadata[idx]
            print(f"\n[{i}] Index: {idx}")
            print(f"    Language: {item['language']}")
            print(f"    ID: {item.get('file_id') or item.get('id')}")
            print(f"    Text: {item.get('transcript') or item.get('text', '')[:80]}...")
            if 'path' in item:
                print(f"    Path: {item['path']}")
    
    def export_results(self):
        """Export search results to CSV"""
        import pandas as pd
        
        query = input("\nEnter search query: ").strip()
        if not query:
            print("❌ Empty query")
            return
        
        top_k = int(input("Number of results (default 100): ") or "100")
        output_file = input("Output filename [search_results.csv]: ").strip() or "search_results.csv"
        
        results = self.system.search_by_text(query, top_k=top_k)
        
        if not results:
            print("❌ No results found")
            return
        
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"✓ Exported {len(results)} results to {output_file}")
    
    def display_results(self, results, title="Search Results"):
        """Display search results"""
        if not results:
            print("\n❌ No results found")
            return
        
        print("\n" + "="*70)
        print(title.upper())
        print("="*70)
        
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] Score: {result['similarity_score']:.4f}")
            print(f"    Language: {result['language']}")
            print(f"    ID: {result.get('file_id') or result.get('id')}")
            
            text = result.get('transcript') or result.get('text', '')
            if text:
                print(f"    Text: {text[:100]}{'...' if len(text) > 100 else ''}")
            
            if 'path' in result:
                print(f"    Path: {result['path']}")
            elif 'audio_file' in result:
                print(f"    File: {result['audio_file']}")
        
        print("\n" + "="*70)
    
    def run(self):
        """Run interactive loop"""
        while True:
            self.display_menu()
            
            try:
                choice = input("\nSelect option: ").strip()
                
                if choice == '1':
                    self.search_text_all()
                elif choice == '2':
                    self.search_text_english()
                elif choice == '3':
                    self.search_text_tamil()
                elif choice == '4':
                    self.search_similar_audio()
                elif choice == '5':
                    self.cross_lingual_en_to_ta()
                elif choice == '6':
                    self.cross_lingual_ta_to_en()
                elif choice == '7':
                    self.view_statistics()
                elif choice == '8':
                    self.browse_samples()
                elif choice == '9':
                    self.export_results()
                elif choice == '0':
                    print("\nExiting... Goodbye!")
                    break
                else:
                    print("❌ Invalid option")
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\nExiting... Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                input("\nPress Enter to continue...")


def main():
    """Main entry point"""
    
    # Import here to avoid circular dependency
    from multilingual_retrieval_system import MultilingualAudioRetrieval
    
    print("\n" + "="*70)
    print("INITIALIZING MULTILINGUAL AUDIO RETRIEVAL SYSTEM")
    print("="*70)
    
    # Initialize system
    retrieval = MultilingualAudioRetrieval()
    
    # Check if unified index exists
    if Path("unified_index/metadata.json").exists():
        print("\nLoading pre-built index...")
        retrieval.load_index("unified_index")
    else:
        print("\nBuilding index from scratch...")
        
        # Load English data
        retrieval.load_english_data()
        
        # Load Tamil data
        retrieval.load_tamil_data()
        
        # Build index
        retrieval.build_unified_index()
        
        # Save for next time
        retrieval.save_index("unified_index")
    
    # Start interactive interface
    interface = InteractiveRetrieval(retrieval)
    interface.run()


if __name__ == "__main__":
    main()
