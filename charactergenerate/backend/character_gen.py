"""
character_gen.py
Core logic for the Character Generation app.
Refactored from CharacterImageGeneration.py with:
  - progress callbacks for SSE streaming
  - get_scenario_summaries() — RAG-derived scenario dropdown
  - generate_prompt_for_scenario() — single call for one scene
"""

import os
import json
import urllib.request
import urllib.parse

from ollama import Client
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

OLLAMA_MODEL = "gemma4:e4b"
OLLAMA_HOST = "http://localhost:11434"


def get_ollama_client() -> Client:
    return Client(host=OLLAMA_HOST)


# ---------------------------------------------------------------------------
# 1. Wikipedia — major character names
# ---------------------------------------------------------------------------

def get_major_character_names(book_title: str) -> list[str]:
    """Query Wikipedia for the book and extract major character names via LLM."""
    try:
        search_query = urllib.parse.quote(book_title)
        search_url = (
            f"https://en.wikipedia.org/w/api.php?action=query&list=search"
            f"&srsearch={search_query}&utf8=&format=json"
        )
        req = urllib.request.Request(search_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            search_data = json.loads(response.read().decode("utf-8"))

        if not search_data["query"]["search"]:
            return []

        page_title = search_data["query"]["search"][0]["title"]
        content_query = urllib.parse.quote(page_title)
        content_url = (
            f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts"
            f"&explaintext=1&titles={content_query}&format=json"
        )
        req2 = urllib.request.Request(content_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req2) as response:
            content_data = json.loads(response.read().decode("utf-8"))

        pages = content_data["query"]["pages"]
        page_id = list(pages.keys())[0]
        content = pages[page_id].get("extract", "")
        if not content:
            return []

        client = get_ollama_client()
        prompt = (
            f"Extract a comma-separated list of major character names from the following "
            f"Wikipedia article about the book '{book_title}'. "
            f"Return ONLY the comma-separated list of names, no other text.\n\n"
            f"Article text:\n{content[:15000]}"
        )
        response = client.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options={"temperature": 0.3, "num_predict": 200},
        )
        text_response = response.response
        major_characters = [n.strip() for n in text_response.split(",") if n.strip()]
        return major_characters

    except Exception as e:
        print(f"[Wikipedia] Error: {e}")
        return []


# ---------------------------------------------------------------------------
# 2. Vector index
# ---------------------------------------------------------------------------

def build_book_index(txt_filepath: str, progress_cb=None, persist_directory: str = None):
    """
    Build (or load) a Chroma vector index for the book.
    progress_cb(stage, n, total, message) is called at each step.
    """
    if persist_directory is None:
        book_name = os.path.splitext(os.path.basename(txt_filepath))[0]
        persist_directory = os.path.join(
            os.path.dirname(os.path.abspath(txt_filepath)),
            f"chroma_db_{book_name}",
        )

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(persist_directory) and any(os.scandir(persist_directory)):
        if progress_cb:
            progress_cb("loading_existing", 1, 1, "Loading existing vector index…")
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    if progress_cb:
        progress_cb("loading_text", 0, 1, "Loading book text…")

    loader = TextLoader(txt_filepath, encoding="utf-8")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    total = len(splits)

    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    batch_size = 50
    for i in range(0, total, batch_size):
        batch = splits[i : i + batch_size]
        vectorstore.add_documents(batch)
        done = min(i + batch_size, total)
        if progress_cb:
            progress_cb("embedding", done, total, f"Embedding chunks {done} / {total}…")

    return vectorstore


# ---------------------------------------------------------------------------
# 3. RAG retrieval
# ---------------------------------------------------------------------------

def get_character_situations(vectorstore, character_name: str, k: int = 6) -> list[str]:
    """Retrieve the top-k chunks most relevant to scenes featuring `character_name`."""
    query = (
        f"Describe a specific scene involving {character_name}, "
        f"including their location, actions, and what they are wearing."
    )
    relevant_docs = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in relevant_docs]


def get_scenario_summaries(vectorstore, character_name: str) -> list[dict]:
    """
    Retrieve k=6 RAG scenes for the character and ask the LLM to distill each
    into a short one-sentence label.
    Returns: [{"label": str, "context": str}, ...]
    """
    situations = get_character_situations(vectorstore, character_name, k=6)
    client = get_ollama_client()
    summaries = []
    for ctx in situations:
        prompt = (
            f"In one short sentence (max 12 words), describe what is happening in this scene "
            f"involving {character_name}. Return ONLY the sentence, nothing else.\n\n"
            f"Scene:\n{ctx[:6000]}"
        )
        resp = client.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options={"temperature": 0.3, "num_predict": 60},
        )
        label = resp.response.strip().strip('"').strip("'")
        summaries.append({"label": label, "context": ctx})
    return summaries


# ---------------------------------------------------------------------------
# 4. Character analysis
# ---------------------------------------------------------------------------

def analyze_character(book_text: str, character_name: str) -> str:
    """Extract a structured character description from a book excerpt."""
    client = get_ollama_client()
    prompt = f"""Analyze the character '{character_name}' from the book. Extract and describe:
- Physical appearance (hair colour, eye colour, height, build, approximate age)
- Clothing and accessories typically worn
- Typical locations where they appear
- Personality traits and mannerisms

Book excerpt:
{book_text[:12000]}"""
    response = client.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        options={"temperature": 0.4, "num_predict": 600, "repeat_penalty": 1.1},
    )
    return response.response


# ---------------------------------------------------------------------------
# 5. Actor casting
# ---------------------------------------------------------------------------

def cast_character_with_actor(
    character_name: str,
    character_description: str,
    industry: str = "hollywood",
    genre: str = "",
    decade: str = "2026"
) -> str:
    """Suggest a real-world actor from the given industry and decade to portray the character in a specific genre."""
    client = get_ollama_client()
    genre_context = f"This is for a {genre} adaptation." if genre else ""
    decade_context = f"The production is set in/filmed during the {decade}s." if decade and decade != "2026" else "The production is modern (2026)."

    prompt = f"""Given the character '{character_name}' with the following description:
{character_description}

{genre_context}
{decade_context}

Cast an age-appropriate real-world actor from the {industry} industry to play this role.
If the decade is in the past, pick an actor who was active and the correct age DURING that decade.
If the decade is modern, pick a currently active actor.
Return ONLY the name of the actor, nothing else."""
    response = client.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        options={"temperature": 0.3, "num_predict": 20},
    )
    return response.response.strip()


# ---------------------------------------------------------------------------
# 6. Prompt generation
# ---------------------------------------------------------------------------

def generate_prompt_for_scenario(
    character_name: str,
    description: str,
    scenario_context: str,
    actor_name: str = "",
    genre: str = "",
    decade: str = "2026",
    gender: str = "",
    race: str = "",
    age: str = "",
) -> str:
    """
    Build a Z-Image-Turbo / Stable Diffusion prompt for one character scene.
    Uses the (possibly user-edited) description and actor name.
    Two-shot approach: draft → critique/refine for richer output.
    """
    client = get_ollama_client()

    # Build override block
    overrides = []
    if gender: overrides.append(f"Gender: {gender}")
    if race:   overrides.append(f"Race/Ethnicity: {race}")
    if age:    overrides.append(f"Age: {age}")
    override_text = "\n".join(overrides)

    # Extract scene-specific details with genre adaptation
    genre_label = genre if genre else "realistic"
    decade_instruction = (
        f"Visual Style: {decade}s cinematography and fashion"
        if decade and decade != "2026"
        else "Visual Style: Modern cinematic hyper-realistic"
    )

    extract_prompt = (
        f"Given this book scene (adapted for the {genre_label} genre), identify the location "
        f"and what {character_name} is doing.\n"
        f"Then describe {character_name}'s CLOTHING and HAIRSTYLE as they would appear in a "
        f"{genre_label} adaptation.\n"
        f"IMPORTANT: Clothing and hair must reflect the {genre_label} genre, but facial features, "
        f"age, and ethnicity must remain consistent with a realistic human portrayal.\n\n"
        f"Scene: {scenario_context[:6000]}"
    )
    scene_resp = client.generate(
        model=OLLAMA_MODEL,
        prompt=extract_prompt,
        options={"temperature": 0.5, "num_predict": 400, "repeat_penalty": 1.1},
    )
    scene_details = scene_resp.response

    actor_instruction = (
        f"Resembles actor: {actor_name}." if actor_name else ""
    )

    # --- Draft prompt (tag/comma style tuned for Z-Image-Turbo) ---
    draft_template = f"""You are an expert at writing Stable Diffusion / Z-Image-Turbo image prompts.

Given the character and scene below, output a single rich image prompt in this EXACT comma-separated format:
[subject], [physical description], [clothing], [action/pose], [setting], [lighting], [mood], [camera angle], [style tags]

Character: {character_name}
Base Description: {description}
{actor_instruction}
Genre: {genre_label}
{decade_instruction}
Scene Details: {scene_details}
{f"Overrides — {override_text}" if override_text else ""}

Rules:
- Comma-separated tags only, no prose sentences
- Be highly specific about lighting (e.g. "golden hour backlight", "dim candlelight", "cool moonlight")
- Include camera angle (e.g. "medium close-up", "low angle shot", "over-the-shoulder")
- Be specific about fabric textures and clothing details
- Include atmospheric background depth
- End with: photorealistic, cinematic, 8k, sharp focus, volumetric lighting
- STRICTLY MAINTAIN: age, ethnicity, and facial structure from the base description
- FULLY ADAPT: clothing, hair, lighting, and environment to match the {genre_label} genre
- Never use character names, book titles, or studio names

Output ONLY the prompt, no explanation."""

    draft_resp = client.generate(
        model=OLLAMA_MODEL,
        prompt=draft_template,
        options={"temperature": 0.75, "num_predict": 600, "top_p": 0.92, "repeat_penalty": 1.1},
    )
    draft = draft_resp.response.strip()

    # --- Refinement pass: critique and enrich the draft ---
    refine_prompt = f"""You are a Z-Image-Turbo prompt expert. Improve this image prompt by making it more visually specific and evocative.

DRAFT PROMPT:
{draft}

Strengthen it by ensuring it includes:
- Exact fabric textures and fine clothing details (e.g. "weathered linen shirt", "embroidered silk collar")
- Specific lighting direction and colour temperature (e.g. "warm amber sidelight from left", "cold blue overcast")
- Background depth with foreground/midground/background elements
- Character's precise expression and body language
- Any environmental atmosphere (dust motes, mist, rain, shadow play)

Return ONLY the improved comma-separated prompt, no prose, no explanation."""

    refined_resp = client.generate(
        model=OLLAMA_MODEL,
        prompt=refine_prompt,
        options={"temperature": 0.6, "num_predict": 700, "top_p": 0.92, "repeat_penalty": 1.1},
    )
    return refined_resp.response.strip()
