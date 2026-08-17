import chromadb
from chromadb.utils import embedding_functions
import json
import re
import numpy as np
import os
from typing import List, Dict, Any, Optional

# Initialize ChromaDB client
client = chromadb.Client()
def load_food_data(file_path: Optional[str] = None) -> List[Dict]:
    """Load food data from JSON file (default: src/FoodDataSet.json).

    Normalizes and ensures the following fields exist on each item:
    - food_id (str), food_name, food_ingredients (list), food_description,
    - cuisine_type, food_calories_per_serving, food_nutritional_factors (dict),
    - food_health_benefits, cooking_method, taste, taste_profile
    """
    if file_path is None:
        # Default to the FoodDataSet.json located in the same directory as this module
        file_path = os.path.join(os.path.dirname(__file__), 'FoodDataSet.json')

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            food_data = json.load(file)

        # Ensure each item has required fields and normalize the structure
        for i, item in enumerate(food_data):
            # Normalize food_id to string
            if 'food_id' not in item:
                item['food_id'] = str(i + 1)
            else:
                item['food_id'] = str(item['food_id'])

            # Basic defaults
            item.setdefault('food_name', f'Food {item["food_id"]}')
            item.setdefault('food_ingredients', [])
            item.setdefault('food_description', '')
            item.setdefault('cuisine_type', 'Unknown')
            item.setdefault('food_calories_per_serving', 0)
            item.setdefault('food_nutritional_factors', {})
            item.setdefault('food_health_benefits', '')
            item.setdefault('cooking_method', '')

            # Merge food_features into top-level keys (appearance, taste, texture, preparation, serving_type, etc.)
            if 'food_features' in item and isinstance(item['food_features'], dict):
                for fk, fv in item['food_features'].items():
                    item[fk] = str(fv) if fv is not None else ''
            else:
                # Ensure common feature keys exist even if food_features missing
                for fk in ['taste', 'texture', 'appearance', 'preparation', 'serving_type']:
                    item.setdefault(fk, '')

            # Keep legacy taste_profile for compatibility (prefer explicit 'taste' when available)
            item['taste_profile'] = item.get('taste_profile', item.get('taste', ''))

        print(f"Successfully loaded {len(food_data)} food items from {file_path}")
        return food_data

    except Exception as e:
        print(f"Error loading food data: {e}")
        return []



