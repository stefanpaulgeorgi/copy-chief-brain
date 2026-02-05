"""
Transcript Ingestion Pipeline

Processes .srt transcript files and stores them in ChromaDB for retrieval.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

import chromadb
from chromadb.config import Settings

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from . import config


def parse_srt(file_path: str) -> List[Dict]:
    """Parse an SRT file into segments with timestamps and text."""
    segments = []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    blocks = re.split(r'\n\n+', content.strip())

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            try:
                seq = int(lines[0].strip())
                timestamp = lines[1].strip()
                text = ' '.join(lines[2:]).strip()

                if text:
                    segments.append({
                        'sequence': seq,
                        'timestamp': timestamp,
                        'text': text
                    })
            except (ValueError, IndexError):
                continue

    return segments


def clean_transcript(segments: List[Dict]) -> str:
    """Convert segments to clean, readable text."""
    texts = [seg['text'] for seg in segments]
    full_text = ' '.join(texts)

    # Clean up
    full_text = re.sub(r'\s+', ' ', full_text)
    full_text = full_text.replace(' ,', ',')
    full_text = full_text.replace(' .', '.')

    return full_text.strip()


def extract_metadata(filename: str, text: str) -> Dict:
    """Extract metadata from filename and content."""
    metadata = {
        'filename': filename,
        'word_count': len(text.split()),
    }

    # Classify niche
    niche_keywords = {
        'ed': ['ed', 'erectile', 'testosterone', 'testolite', 'balls', 'bedroom', 'libido'],
        'weight_loss': ['weight', 'fat', 'diet', 'brazilian', 'slimming', 'metabolism'],
        'supplements': ['supplement', 'vitamin', 'nutrient', 'fulvic', 'shilajit'],
        'vision': ['vision', 'eye', 'sight', 'macular', 'glasses'],
        'golf': ['golf', 'swing', 'shaft', 'drive', 'handicap'],
        'hair': ['hair', 'bald', 'follicle', 'scalp'],
        'skin': ['skin', 'wrinkle', 'nail', 'fungus', 'aging'],
        'dog': ['dog', 'pet', 'canine', 'anxiety'],
        'brain': ['brain', 'cognitive', 'memory', 'fog'],
    }

    combined_text = (filename + ' ' + text[:2000]).lower()

    for niche, keywords in niche_keywords.items():
        if any(kw in combined_text for kw in keywords):
            metadata['niche'] = niche
            break
    else:
        metadata['niche'] = 'general'

    return metadata


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """Split text into overlapping chunks."""
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap

        # Prevent infinite loop for small texts
        if end >= len(words):
            break

    return chunks


def get_embedding(text: str, client: openai.OpenAI) -> List[float]:
    """Get embedding for a piece of text using OpenAI."""
    response = client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def process_transcript(file_path: Path) -> Dict:
    """Process a single transcript file."""
    segments = parse_srt(str(file_path))
    clean_text = clean_transcript(segments)
    metadata = extract_metadata(file_path.name, clean_text)
    chunks = chunk_text(clean_text)

    return {
        'filename': file_path.name,
        'metadata': metadata,
        'full_text': clean_text,
        'chunks': chunks,
    }


def ingest_transcripts(
    transcript_dir: str,
    clear_existing: bool = False
) -> Dict:
    """
    Process all transcripts and store in ChromaDB.

    Args:
        transcript_dir: Path to directory containing .srt files
        clear_existing: Whether to clear existing data before ingesting

    Returns:
        Summary of ingestion results
    """
    transcript_path = Path(transcript_dir)

    if not transcript_path.exists():
        raise ValueError(f"Directory not found: {transcript_dir}")

    # Initialize OpenAI client
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set. Add it to your .env file.")

    openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False)
    )

    # Get or create collection
    if clear_existing:
        try:
            chroma_client.delete_collection(config.COLLECTION_NAME)
            print(f"Cleared existing collection: {config.COLLECTION_NAME}")
        except:
            pass

    collection = chroma_client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"description": "Stefan's copy review transcripts"}
    )

    # Find all .srt files
    srt_files = list(transcript_path.glob("*.srt"))
    print(f"Found {len(srt_files)} .srt files")

    if not srt_files:
        # Also check for .txt files (processed transcripts)
        txt_files = list(transcript_path.glob("*.txt"))
        if txt_files:
            print(f"Found {len(txt_files)} .txt files instead")

    stats = {
        'files_processed': 0,
        'chunks_added': 0,
        'total_words': 0,
        'by_niche': {},
        'errors': []
    }

    for file_path in tqdm(srt_files, desc="Processing transcripts"):
        try:
            # Process transcript
            result = process_transcript(file_path)

            # Generate embeddings and add to ChromaDB
            for i, chunk in enumerate(result['chunks']):
                chunk_id = f"{file_path.stem}_{i:03d}"

                # Get embedding
                embedding = get_embedding(chunk, openai_client)

                # Add to collection
                collection.add(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{
                        'filename': result['filename'],
                        'niche': result['metadata']['niche'],
                        'chunk_index': i,
                        'total_chunks': len(result['chunks']),
                    }]
                )

                stats['chunks_added'] += 1

            stats['files_processed'] += 1
            stats['total_words'] += result['metadata']['word_count']

            niche = result['metadata']['niche']
            stats['by_niche'][niche] = stats['by_niche'].get(niche, 0) + 1

        except Exception as e:
            stats['errors'].append({
                'file': file_path.name,
                'error': str(e)
            })
            print(f"\nError processing {file_path.name}: {e}")

    print(f"\n=== Ingestion Complete ===")
    print(f"Files processed: {stats['files_processed']}")
    print(f"Chunks added: {stats['chunks_added']}")
    print(f"Total words: {stats['total_words']:,}")
    print(f"By niche: {stats['by_niche']}")

    if stats['errors']:
        print(f"Errors: {len(stats['errors'])}")

    return stats


def ingest_jsonl(jsonl_path: str, clear_existing: bool = False) -> Dict:
    """
    Ingest from a pre-processed JSONL file.

    Args:
        jsonl_path: Path to the JSONL file
        clear_existing: Whether to clear existing data

    Returns:
        Summary of ingestion results
    """
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set. Add it to your .env file.")

    openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False)
    )

    if clear_existing:
        try:
            chroma_client.delete_collection(config.COLLECTION_NAME)
            print(f"Cleared existing collection: {config.COLLECTION_NAME}")
        except:
            pass

    collection = chroma_client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"description": "Stefan's copy review transcripts"}
    )

    stats = {
        'files_processed': 0,
        'chunks_added': 0,
        'total_words': 0,
        'by_niche': {},
        'errors': []
    }

    # Read JSONL
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Found {len(lines)} transcripts in JSONL")

    for line in tqdm(lines, desc="Processing transcripts"):
        try:
            record = json.loads(line)

            # Chunk the text
            chunks = chunk_text(record['text'])

            # Generate embeddings and add to ChromaDB
            for i, chunk in enumerate(chunks):
                chunk_id = f"{Path(record['filename']).stem}_{i:03d}"

                embedding = get_embedding(chunk, openai_client)

                collection.add(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{
                        'filename': record['filename'],
                        'niche': record.get('niche', 'general'),
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                    }]
                )

                stats['chunks_added'] += 1

            stats['files_processed'] += 1
            stats['total_words'] += record.get('word_count', len(record['text'].split()))

            niche = record.get('niche', 'general')
            stats['by_niche'][niche] = stats['by_niche'].get(niche, 0) + 1

        except Exception as e:
            stats['errors'].append({
                'line': line[:100],
                'error': str(e)
            })

    print(f"\n=== Ingestion Complete ===")
    print(f"Files processed: {stats['files_processed']}")
    print(f"Chunks added: {stats['chunks_added']}")
    print(f"Total words: {stats['total_words']:,}")
    print(f"By niche: {stats['by_niche']}")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest transcripts into ChromaDB")
    parser.add_argument("--input", "-i", required=True, help="Path to transcripts dir or JSONL file")
    parser.add_argument("--clear", action="store_true", help="Clear existing data before ingesting")

    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.suffix == '.jsonl':
        ingest_jsonl(str(input_path), clear_existing=args.clear)
    else:
        ingest_transcripts(str(input_path), clear_existing=args.clear)
