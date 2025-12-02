# app/services.py
from pathlib import Path
from typing import List

import requests
import re
import shutil
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

# -------------------------------------------------------------------
# Setup: paths, env, OpenAI client
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
IMAGES_DIR = STATIC_DIR / "images"
AUDIO_DIR = STATIC_DIR / "audio"

# Make sure folders exist
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Load .env and create a single client for reuse
load_dotenv(find_dotenv(), override=True)
client = OpenAI()


def safe_filename(name: str) -> str:
    """
    Convert title to a filesystem-safe filename:
    - remove * and other invalid symbols
    - replace spaces with underscores
    - allow letters, numbers, underscores, and hyphens only
    """
    name = name.strip()
    name = name.replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9_\-]", "", name)
    return name

# -------------------------------------------------------------------
# 1. Meal plan generation (your create_meals)
# -------------------------------------------------------------------
def create_meals(
    ingredients: str,
    kcal: int = 2000,
    exact_ingredients: bool = False,
    output_format: str = "text",
    model: str = "gpt-3.5-turbo",
    system_role: str = "You are a skilled cook with the expertise of a chef.",
    temperature: float = 1.0,
    extra: str | None = None,
) -> str:
    """
    Use GPT to generate a full-day meal plan (breakfast, lunch, dinner)
    as one big text block. Last line must contain only the recipe titles,
    comma-separated. Recipes are separated by 50 dashes.
    """
    prompt = f"""
    Create a healthy daily meal plan for breakfast, lunch, and dinner based on the following ingredients: {ingredients}.
    Your output should be in the {output_format} format.
    Follow the instructions below carefully.
    ### Instructions:
    1. {'Use ONLY the provided ingredients with salt, pepper, and spices.' if exact_ingredients else 'Feel free to incorporate the provided ingredients as a base and add other ingredients if you consider them necessary to enhance the flavor, nutritional value, or overall appeal of the recipes.'}  
    2. Specify the exact amount of each ingredient.
    3. Ensure that the total daily calorie intake is below {kcal}. 
    4. For each meal, explain each recipe, step by step, in clear and simple sentences. Use bullet points or numbers to organize the steps.
    5. For each meal, specify the total number of calories and the number of servings.
    6. For each meal, provide a concise and descriptive title that summarizes the main ingredients and flavors. The title should also be a valid DALL-E prompt to generate an original image for the meal.
    7. For each recipe, indicate the prep, cook and total time.
    {'8. If possible the meals should be:' + extra if extra else '8. '}
    9. Separate the recipes with 50 dashes (-).
    10. Do NOT use any backticks (`) or code fences anywhere in your answer. This includes single backticks and triple backticks.
    
    Before answering, make sure that you have followed the instructions listed above (points 1 to 10).
    The last line of your answer should be a string that contains ONLY the titles of the recipes and nothing more with a comma in between.
    Example of the last line of your answer: 
    '\\nBroccoli and Egg Scramble, Grilled Chicken and Vegetable, Baked Fish with Cabbage Slaw'.  
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_role},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content


# -------------------------------------------------------------------
# Helpers: split plan into meals + extract titles
# -------------------------------------------------------------------
def split_meals(plan_text: str) -> List[str]:
    """
    Split the full GPT response into individual meal sections.

    Instead of requiring exactly 50 dashes, treat any line that is
    mostly dashes (length >= 10) as a separator.
    """
    lines = plan_text.splitlines()
    blocks: list[list[str]] = []
    current: list[str] = []

    for line in lines:
        stripped = line.strip()
        # separator = a line that is only dashes, length >= 10
        if stripped and set(stripped) == {"-"} and len(stripped) >= 10:
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(line)

    if current:
        blocks.append(current)

    # Join each block back into a text chunk
    return ["\n".join(b).strip() for b in blocks if any(l.strip() for l in b)]



def extract_titles(plan_text: str) -> List[str]:
    """
    Take the last line of the GPT response and split by commas
    to recover the recipe titles.
    """
    lines = plan_text.splitlines()
    if not lines:
        return []
    titles_line = lines[-1].strip()
    titles = [t.strip(" '") for t in titles_line.split(",") if t.strip()]
    return titles


# -------------------------------------------------------------------
# 2. Image generation (your create_and_save_image)
# -------------------------------------------------------------------
def create_and_save_image(
    title: str,
    model: str = "dall-e-3",
    size: str = "1024x1024",
    quality: str = "standard",
    extra: str = "",
) -> str:
    """
    Generate an image for a meal title using DALL-E, save it under
    app/static/images/, and return the full filesystem path.
    """
    image_prompt = (
        f"{title}, hd quality, "
        "Top-down view of the entire dish, fully visible, centered in the image, "
        "on a plain white background, no cropping., "
        f"{extra}"
    )

    response = client.images.generate(
        model=model,
        prompt=image_prompt,
        style="natural",
        size=size,
        quality=quality,
    )

    image_url = response.data[0].url

    # Simple filename based on title
    safe_title = safe_filename(title)
    image_path = IMAGES_DIR / f"{safe_title}.png"
    image_resource = requests.get(image_url, stream=True)
    if image_resource.status_code == 200:
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image_resource.raw, f)
        return str(image_path)
    else:
        raise RuntimeError(f"Error downloading image from {image_url}")


# -------------------------------------------------------------------
# 3. TTS: make spoken text + save MP3 (your speak, plus the GPT clean-up)
# -------------------------------------------------------------------
def make_spoken_recipe_text(recipe: str, model: str = "gpt-3.5-turbo") -> str:
    """
    Use GPT to lightly rewrite the recipe so it sounds better when read aloud.
    Matches the logic from your notebook.
    """
    prompt = (
        "I will provide a recipe that will be spoken aloud.\n"
        "If necessary, you will adjust the recipe to make it more readable.\n"
        "You can also add an introduction and summary, but you won't change the "
        "ingredients, instructions, or other key elements of the recipe.\n\n"
        f"Recipe:\n{recipe}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def speak(recipe: str, filename_prefix: str) -> str:
    """
    Generate TTS audio for a recipe:
      1) Use GPT to make it more 'speakable'
      2) Use tts-1 to turn it into MP3
      3) Save to app/static/audio/{filename_prefix}.mp3

    Returns the full filesystem path to the MP3.
    """
    spoken_recipe = make_spoken_recipe_text(recipe)

    audio_path = AUDIO_DIR / f"{filename_prefix}.mp3"

    response = client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=spoken_recipe,
    )
    response.stream_to_file(str(audio_path))

    return str(audio_path)
