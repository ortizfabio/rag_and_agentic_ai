from shared_functions import *

# Define the embedding function using SentenceTransformers
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

# Create a new instance of ChromaClient to interact with the Chroma DB
client = chromadb.Client()

# Global variable to store loaded food items
food_items = []


def create_similarity_search_collection(name, description):
    return client.create_collection(
        name=name,
        metadata={"description": "A collection for storing food receipt"},
        configuration={
            "hnsw": {"space": "cosine"},
            "embedding_function": ef }
    )


def populate_similarity_collection(collection, food_items):
    """Populate Chroma collection with documents and rich metadata from food items."""
    food_documents = []
    metadata = []
    food_ids = []
    ids = set()

    for item in food_items:
        # Safely process each item; if one item is malformed, skip it and continue
        try:
            if item["food_id"] not in ids:
                food_nutritional_factors = item.get("food_nutritional_factors", {})
                cuisine = item.get('cuisine_type', 'Unknown')
                ingredients = item.get('food_ingredients', []) or []
                description = item.get('food_description', '')
                health = item.get('food_health_benefits', '')
                cooking = item.get('cooking_method', '')
                taste = item.get('taste', '')
                texture = item.get('texture', '')
                appearance = item.get('appearance', '')
                preparation = item.get('preparation', '')
                serving_type = item.get('serving_type', '')

                # Build a human-readable document for embedding
                doc_parts = [f"{item.get('food_name', 'Unknown')}"]
                if cuisine:
                    doc_parts.append(f"({cuisine})")
                doc_parts.append(f"has {item.get('food_calories_per_serving', 0)} calories.")
                if description:
                    doc_parts.append(description)
                if ingredients:
                    doc_parts.append("Ingredients: " + ", ".join(ingredients))
                if taste:
                    doc_parts.append("Taste: " + taste)
                if cooking:
                    doc_parts.append("Cooking method: " + cooking)
                if food_nutritional_factors:
                    cf = food_nutritional_factors
                    carb = cf.get('carbohydrates', '')
                    prot = cf.get('protein', '')
                    fat = cf.get('fat', '')
                    doc_parts.append(f"Nutrition: Carbs {carb}, Protein {prot}, Fat {fat}.")

                document = " ".join(doc_parts)
                food_documents.append(document)

                metadata.append({
                    "name": item.get('food_name'),
                    "calories": item.get('food_calories_per_serving'),
                    "carbohydrates": food_nutritional_factors.get('carbohydrates', ''),
                    "protein": food_nutritional_factors.get('protein', ''),
                    "fat": food_nutritional_factors.get('fat', ''),
                    "cuisine_type": cuisine,
                    #"ingredients": ingredients,
                    "description": description,
                    "food_health_benefits": health,
                    "cooking_method": cooking,
                    "taste": taste,
                    "texture": texture,
                    "appearance": appearance,
                    "preparation": preparation,
                    "serving_type": serving_type
                })

                food_ids.append(item['food_id'])
                ids.add(item['food_id'])
        except Exception as e:
            # Log and skip problematic item
            try:
                fid = item.get('food_id', '(unknown)')
            except Exception:
                fid = '(unreadable)'
            print(f"Warning: skipping food item {fid} due to error: {e}")
            continue
    try:
        # Add data to the collection
        collection.add(
            ids=food_ids,
            documents=food_documents,
            metadatas=metadata
        )
    except Exception as e:
        print(f"Error adding data to collection: {e}")
        raise e


def main():
    """Main function for interactive CLI food recommendation system"""
    try:
        print("🍽️  Interactive Food Recommendation System")
        print("=" * 50)
        print("Loading food database...")

        # Load food data from file
        global food_items
        food_items = load_food_data()
        print(f"✅ Loaded {len(food_items)} food items successfully")

        # Create and populate search collection
        collection = create_similarity_search_collection(
            "interactive_food_search",
            {'description': 'A collection for interactive food search'}
        )
        populate_similarity_collection(collection, food_items)

        # Start interactive chatbot
        interactive_food_chatbot(collection)

    except Exception as error:
        print(f"❌ Error initializing system: {error}")


def interactive_food_chatbot(collection):
    """Interactive CLI chatbot for food recommendations"""
    print("\n" + "=" * 50)
    print("🤖 INTERACTIVE FOOD SEARCH CHATBOT")
    print("=" * 50)
    print("Commands:")
    print("  • Type any food name or description to search")
    print("  • 'help' - Show available commands")
    print("  • 'quit' or 'exit' - Exit the system")
    print("  • Ctrl+C - Emergency exit")
    print("-" * 50)

    while True:
        try:
            # Get user input
            user_input = input("\n🔍 Search for food: ").strip()

            # Handle empty input
            if not user_input:
                print("   Please enter a search term or 'help' for commands")
                continue

            # Handle exit commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using the Food Recommendation System!")
                print("   Goodbye!")
                break

            # Handle help command
            elif user_input.lower() in ['help', 'h']:
                show_help_menu()

            # Handle food search
            else:
                handle_food_search(collection, user_input)

        except KeyboardInterrupt:
            print("\n\n👋 System interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error processing request: {e}")


def show_help_menu():
    """Display help information for users"""
    print("\n📖 HELP MENU")
    print("-" * 30)
    print("Search Examples:")
    print("  • 'chocolate dessert' - Find chocolate desserts")
    print("  • 'Italian food' - Find Italian cuisine")
    print("  • 'sweet treats' - Find sweet desserts")
    print("  • 'baked goods' - Find baked items")
    print("  • 'low calorie' - Find lower-calorie options")
    print("\nCommands:")
    print("  • 'help' - Show this help menu")
    print("  • 'quit' - Exit the system")


def perform_similarity_search(collection, query_term, limit):
    try:
        # Define the query term you want to search for in the collection

        # Perform a query to search for the most similar documents to the 'query_term'
        results = collection.query(
            query_texts=[query_term],
            n_results=limit  # Retrieve top 5 results
        )
        return results
    except Exception as error:
        print(f"Error in similarity search: {error}")


def handle_food_search(collection, query):
    """Handle food similarity search with enhanced display"""
    print(f"\n🔍 Searching for '{query}'...")
    print("   Please wait...")

    # Perform similarity search
    results = perform_similarity_search(collection, query, 5)

    if not results or not results.get('distances'):
        print("❌ No matching foods found.")
        print("💡 Try different keywords like:")
        print("   • Cuisine types: 'Italian', 'American'")
        print("   • Ingredients: 'chocolate', 'flour', 'cheese'")
        print("   • Descriptors: 'sweet', 'baked', 'dessert'")
        return

    # Chroma returns nested lists per query: take first query's results
    metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
    documents = results.get('documents', [[]])[0] if results.get('documents') else []
    distances = results.get('distances', [[]])[0] if results.get('distances') else []
    ids = results.get('ids', [[]])[0] if results.get('ids') else []

    processed = []
    for meta, doc, dist, _id in zip(metadatas, documents, distances, ids):
        # meta may be a dict or None
        name = meta.get('name') if isinstance(meta, dict) else None
        calories = meta.get('calories') if isinstance(meta, dict) else None
        cuisine = meta.get('cuisine_type') if isinstance(meta, dict) else None
        description = meta.get('description') if isinstance(meta, dict) else (doc or '')
        ingredients = meta.get('ingredients') if isinstance(meta, dict) else []
        health_benefits = meta.get('food_health_benefits') if isinstance(meta, dict) else ''
        cooking_method = meta.get('cooking_method') if isinstance(meta, dict) else ''
        taste = meta.get('taste') if isinstance(meta, dict) else ''
        texture = meta.get('texture') if isinstance(meta, dict) else ''
        appearance = meta.get('appearance') if isinstance(meta, dict) else ''
        preparation = meta.get('preparation') if isinstance(meta, dict) else ''
        serving_type = meta.get('serving_type') if isinstance(meta, dict) else ''
        similarity = max(0.0, 1.0 - float(dist))  # distance -> similarity for cosine
        processed.append({
            'id': _id,
            'name': name,
            'calories': calories,
            'cuisine_type': cuisine,
            'description': description,
            'ingredients': ingredients,
            'food_health_benefits': health_benefits,
            'cooking_method': cooking_method,
            'taste': taste,
            'texture': texture,
            'appearance': appearance,
            'preparation': preparation,
            'serving_type': serving_type,
            'similarity': similarity
        })

    if not processed:
        print("❌ No matching foods found.")
        return

    print(f"\n✅ Found {len(processed)} recommendations:")
    print("=" * 60)

    for i, item in enumerate(processed, 1):
        percentage_score = item['similarity'] * 100
        print(f"\n{i}. 🍽️  {item.get('name') or 'Unknown'}")
        print(f"   📊 Match Score: {percentage_score:.1f}%")
        print(f"   🏷️  Cuisine: {item.get('cuisine_type') or 'Unknown'}")
        print(f"   🔥 Calories: {item.get('calories') or 'Unknown'} per serving")
        print(f"   📝 Description: {item.get('description')}")
        if item.get('ingredients'):
            print(f"   🧾 Ingredients: {', '.join(item.get('ingredients'))}")
        if item.get('food_health_benefits'):
            print(f"   ❤️ Health Benefits: {item.get('food_health_benefits')}")
        if item.get('cooking_method'):
            print(f"   🍳 Cooking: {item.get('cooking_method')}")
        if item.get('taste'):
            print(f"   👅 Taste: {item.get('taste')}")
        if item.get('texture'):
            print(f"   🧩 Texture: {item.get('texture')}")
        if item.get('appearance'):
            print(f"   👀 Appearance: {item.get('appearance')}")
        if item.get('preparation'):
            print(f"   🛠️ Preparation: {item.get('preparation')}")
        if item.get('serving_type'):
            print(f"   🍽️ Serving: {item.get('serving_type')}")

        if i < len(processed):
            print("   " + "-" * 50)

    print("=" * 60)

    # Provide suggestions for further exploration
    suggest_related_searches(processed)


def suggest_related_searches(results):
    """Suggest related searches based on current results"""
    if not results:
        return

    # Results is a list of dicts with keys: name, calories, cuisine_type, description, similarity
    cuisines = list({r.get('cuisine_type') for r in results if r.get('cuisine_type')})

    print("\n💡 Related searches you might like:")
    for cuisine in cuisines[:3]:  # Limit to 3 suggestions
        print(f"   • Try '{cuisine} dishes' for more {cuisine} options")

    # Suggest calorie-based searches
    cal_values = [r.get('calories') for r in results if isinstance(r.get('calories'), (int, float))]
    if cal_values:
        avg_calories = sum(cal_values) / len(cal_values)
        if avg_calories > 350:
            print("   • Try 'low calorie' for lighter options")
        else:
            print("   • Try 'hearty meal' for more substantial dishes")




if __name__ == "__main__":
    main()
