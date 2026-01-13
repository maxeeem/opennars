import os
import shutil
import markdown

PAPER_DIR = "paper"
ASSETS_DIR = os.path.join(PAPER_DIR, "paper_assets")
FIG_DIR = "docs/figures"

def main():
    # 1. Copy figures
    if not os.path.exists(FIG_DIR):
        print("No figures found. Run scripts/make_figures.py first.")
        return

    os.makedirs(ASSETS_DIR, exist_ok=True)
    for fn in os.listdir(FIG_DIR):
        if fn.endswith(".png"):
            shutil.copy(os.path.join(FIG_DIR, fn), os.path.join(ASSETS_DIR, fn))
            print(f"Copied {fn}")

    # 2. Convert MD to HTML
    with open(os.path.join(PAPER_DIR, "paper.md"), "r") as f:
        text = f.read()
    
    html = markdown.markdown(text)
    
    # Wrap in simple styling
    html_full = f"""
    <html>
    <head>
        <style>
            body {{ font-family: sans-serif; max-width: 800px; margin: 40px auto; }}
            img {{ max-width: 100%; }}
        </style>
    </head>
    <body>
    {html}
    </body>
    </html>
    """
    
    out_path = os.path.join(PAPER_DIR, "paper.html")
    with open(out_path, "w") as f:
        f.write(html_full)
    
    print(f"Generated {out_path} (Print this to PDF)")

if __name__ == "__main__":
    try:
        import markdown
    except ImportError:
        print("Please install markdown: pip install markdown")
        exit(1)
    main()
