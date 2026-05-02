import subprocess

MODEL = "llama3:8b"

def refine_prompt(user_request: str) -> str:
    """
    Turns a simple user request into a strong photorealistic outpainting prompt.
    Requires: ollama installed + model pulled (ollama pull llama3:8b)
    """
    system = (
         "You write ONE-LINE prompts for PHOTO-REALISTIC portrait outpainting/inpainting.\n"
         "MOST IMPORTANT: Follow the user's request exactly; do not add new changes not requested.\n"
         "Rules:\n"
         "- Keep the same person/identity.\n"
         "- Preserve face shape and facial features; do not stylize.\n"
         "- Natural hair texture and realistic lighting.\n"
         "- Avoid cartoon, anime, painting.\n"
         "- Keep background consistent unless user asks to change it.\n"
         "- Output ONLY ONE line prompt.\n"
)

    full = f"{system}\nUser request: {user_request}\nFinal prompt:"
    p = subprocess.run(
        ["ollama", "run", MODEL],
        input=full.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False
    )
    out = p.stdout.decode("utf-8", errors="ignore").strip()
    lines = [x.strip() for x in out.splitlines() if x.strip()]
    return lines[-1] if lines else user_request
