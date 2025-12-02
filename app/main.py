# app/main.py
from pathlib import Path
from typing import List

from markdown import markdown

import os

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, HttpUrl

from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse


from .services import (
    create_meals,
    split_meals,
    extract_titles,
    create_and_save_image,
    speak,
)

templates = Jinja2Templates(directory="app/templates")

# ---------------------------------------------------------
# Paths & FastAPI app
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="AI Daily Meal Planner",
    version="1.0.0",
    description="Generates a 3-meal daily plan with images and audio.",
)

# Serve static files (images + audio)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------
class MealPlanRequest(BaseModel):
    ingredients: str = Field(
        ...,
        description="Comma-separated list of base ingredients."
    )
    kcal: int = Field(
        2000,
        description="Maximum total calories for the day."
    )
    exact_ingredients: bool = Field(
        False,
        description="If true, use only the provided ingredients (plus salt/pepper/spices)."
    )
    extra: str | None = Field(
        default=None,
        description="Extra preference, e.g. 'spicy', 'high protein', etc."
    )


class Meal(BaseModel):
    title: str
    text: str
    image_url: HttpUrl
    audio_url: HttpUrl
    html_url: HttpUrl


class MealPlanResponse(BaseModel):
    raw_plan: str
    meals: List[Meal]


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def path_to_url(path: str | Path, request: Request) -> str:
    """
    Convert a filesystem path under app/static into a public URL.
    Example: C:/.../app/static/images/foo.png -> http://host/static/images/foo.png
    """
    path = Path(path).resolve()
    static_root = STATIC_DIR.resolve()

    # path relative to app/static (e.g. "images/foo.png")
    rel = path.relative_to(static_root)
    rel_url = f"static/{rel.as_posix()}"

    # request.base_url ends with "/", so this gives ".../static/images/foo.png"
    return str(request.base_url) + rel_url


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "AI Daily Meal Planner API. POST to /meal_plan."}


@app.get("/meal_plan_html/{meal_name}", response_class=HTMLResponse)
async def get_meal_html(meal_name: str, request: Request):
    valid = ["breakfast", "lunch", "dinner"]
    if meal_name.lower() not in valid:
        raise HTTPException(400, "Meal must be breakfast, lunch, or dinner")

    meal_name = meal_name.lower()

    # read markdown-style text
    filename = f"{meal_name}.txt"
    filepath = os.path.join("app", "static", "recipes", filename)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except FileNotFoundError:
        raise HTTPException(404, "Meal HTML not generated yet")

    html_body = markdown(raw_text)

    # build full URLs for image + audio
    base = str(request.base_url).rstrip("/")
    image_url = f"{base}/static/images/{meal_name}.png"
    audio_url = f"{base}/static/audio/{meal_name}.mp3"

    return templates.TemplateResponse(
        "meal_template.html",
        {
            "request": request,
            "title": meal_name.capitalize(),
            "body": html_body,
            "image_url": image_url,
            "audio_url": audio_url,
        },
    )


@app.post("/meal_plan", response_model=MealPlanResponse)
async def generate_meal_plan(payload: MealPlanRequest, request: Request):
    """
    Generate a 3-meal daily plan, plus an image and TTS audio for each meal.
    """

    # 1. Call GPT to generate the full-day plan text
    plan_text = create_meals(
        ingredients=payload.ingredients,
        kcal=payload.kcal,
        exact_ingredients=payload.exact_ingredients,
        output_format="text",
        extra=payload.extra,
    )

    if not plan_text or not plan_text.strip():
        raise HTTPException(status_code=500, detail="Meal plan generation failed.")

    html_text_dir = os.path.join(STATIC_DIR, "recipes")
    os.makedirs(html_text_dir, exist_ok=True)


    # 2. Split into 3 meal sections and extract titles from last line
    meal_sections = split_meals(plan_text)      # text blocks separated by 50 dashes
    titles = extract_titles(plan_text)          # last line, comma-separated

    if len(meal_sections) < 3 or len(titles) < 3:
        # For now, just log what happened and use as many as we have
        print("DEBUG split_meals:", len(meal_sections), "sections")
        print("DEBUG extract_titles:", len(titles), "titles")
        # Use the minimum count we have (up to 3)
        count = min(3, len(meal_sections), len(titles))
        if count == 0:
            raise HTTPException(
                status_code=500,
                detail="Meal plan from GPT was empty or badly formatted.",
            )
        meal_sections = meal_sections[:count]
        titles = titles[:count]

    # Only use the first 3 to keep it predictable
    meal_sections = meal_sections[:3]
    titles = titles[:3]

    meals: List[Meal] = []

    meal_names = ["breakfast", "lunch", "dinner"]
    meals: list[Meal] = []

    for idx, (section_text, title) in enumerate(zip(meal_sections, titles), start=0):
        meal_name = meal_names[idx]

        # save text for HTML
        text_path = os.path.join(html_text_dir, f"{meal_name}.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(section_text)

        # image: now saved as breakfast.png / lunch.png / dinner.png
        image_path = create_and_save_image(
            title,
            extra="white background",
            filename_prefix=meal_name,
        )
        image_url = path_to_url(image_path, request)

        # audio: already using meal_name
        audio_path = speak(section_text, filename_prefix=meal_name)
        audio_url = path_to_url(audio_path, request)

        html_url = f"{request.base_url}meal_plan_html/{meal_name}"

        meals.append(
            Meal(
                title=title,
                text=section_text,
                image_url=image_url,
                audio_url=audio_url,
                html_url=html_url,
            )
        )


    # 4. Return full raw text + structured meals
    return MealPlanResponse(
        raw_plan=plan_text,
        meals=meals,
    )

