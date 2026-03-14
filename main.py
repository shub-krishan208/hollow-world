import subprocess
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_path>")
        sys.exit(1)

    prompt = "Tell me the name of the object in this image."
    image = sys.argv[1]

    full_prompt = f"{prompt} {image}"

    result = subprocess.run(
        ["ollama", "run", "llava", full_prompt],
        capture_output=True,
        text=True
    )

    print(result.stdout)


if __name__ == "__main__":
    main()
