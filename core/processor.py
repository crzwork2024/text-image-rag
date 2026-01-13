import re
import uuid
import json
import hashlib
import logging
from typing import List, Dict, Tuple
from config import config

logger = logging.getLogger(__name__)

def get_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def process_markdown_to_chunks(md_text: str) -> Tuple[List[Dict], Dict[str, str]]:
    """Splits markdown into parent sections and child paragraphs."""
    sections = re.split(r'\n(?=#+ )', md_text)
    vector_items = []
    parent_map = {}

    for section in sections:
        section = section.strip()
        if not section: continue
        
        section_hash = get_hash(section)
        parent_map[section_hash] = section
        
        paragraphs = [p.strip() for p in section.split('\n\n') if p.strip()]
        for para in paragraphs:
            if para.startswith('#'): continue
            
            vector_items.append({
                "id": str(uuid.uuid4()),
                "text": para,
                "metadata": {"parent_hash": section_hash}
            })

    # Debug exports
    logger.info(f"Exporting debug chunks to {config.DEBUG_EXPORT_PATH}")
    with open(config.DEBUG_EXPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(vector_items, f, ensure_ascii=False, indent=2)
        
    return vector_items, parent_map