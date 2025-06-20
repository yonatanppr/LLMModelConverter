import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def compile_package_with_coremlc(input_path: str, output_path: str) -> None:
    """
    Compiles a .mlpackage to a .mlmodelc using the 'coremlc' command-line tool.
    This is the most reliable method and requires Xcode Command Line Tools.

    Args:
        input_path (str): Path to the source .mlpackage directory.
        output_path (str): Path where the compiled .mlmodelc directory will be saved.
    """
    input_path_p = Path(input_path).resolve()
    output_path_p = Path(output_path).resolve()

    # --- Input Validation ---
    if not input_path_p.exists() or not str(input_path_p).endswith(".mlpackage"):
        print(f"Error: Input path '{input_path_p}' is not a valid .mlpackage directory.", file=sys.stderr)
        sys.exit(1)

    # --- Pre-flight Checks ---
    if shutil.which("xcrun") is None:
        print("Error: 'xcrun' command not found.", file=sys.stderr)
        print("Please ensure Xcode Command Line Tools are installed (`xcode-select --install`)", file=sys.stderr)
        sys.exit(1)

    # --- Directory Management ---
    # The `coremlc` tool requires the output directory to exist.
    # It places the compiled model *inside* this directory.
    # We will compile into a temporary location first to control the final naming.
    temp_output_dir = output_path_p.parent / f"temp_compile_{output_path_p.stem}"

    # Clean up from any previous failed runs
    if temp_output_dir.exists():
        shutil.rmtree(temp_output_dir)
    if output_path_p.exists():
        shutil.rmtree(output_path_p)

    temp_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Compiling '{input_path_p.name}' using 'xcrun coremlc'...")

    try:
        # --- Build and Execute the Command ---
        command = [
            "xcrun", "coremlc", "compile",
            str(input_path_p),
            str(temp_output_dir)
        ]

        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True  # Raises CalledProcessError if the command returns a non-zero exit code
        )
        print("Compilation successful.")

        # --- Find and Move the Compiled Model ---
        # The compiled model will be inside our temp directory.
        # e.g., 'temp_compile_output/ClerkT5Summarizer.mlmodelc'
        compiled_model_name = input_path_p.with_suffix('.mlmodelc').name
        compiled_model_path = temp_output_dir / compiled_model_name

        if compiled_model_path.exists():
            print(f"Moving compiled model to final destination: {output_path_p}")
            shutil.move(str(compiled_model_path), str(output_path_p))
            print("✅ Conversion complete.")
        else:
            print(f"Error: Compiled model '{compiled_model_name}' not found in the output directory.", file=sys.stderr)
            print(f"Check the output of coremlc for details.", file=sys.stderr)
            sys.exit(1)

    except subprocess.CalledProcessError as e:
        print("❌ Error: Compilation with 'coremlc' failed.", file=sys.stderr)
        print("------------------- coremlc stderr -------------------", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        print("------------------------------------------------------", file=sys.stderr)
        sys.exit(1)
    finally:
        # --- Cleanup ---
        if temp_output_dir.exists():
            shutil.rmtree(temp_output_dir)


if __name__ == "__main__":
    # Modify these variables to test directly in an IDE
    default_input = "ClerkT5Summarizer.mlpackage"
    default_output = "LocalLLM.mlmodelc"

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Compile a .mlpackage to a .mlmodelc for on-device deployment.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input", nargs='?', default=default_input, help="Path to the source .mlpackage directory.")
    parser.add_argument("output", nargs='?', default=default_output,
                        help="Destination path for the compiled .mlmodelc directory.")
    args = parser.parse_args()

    compile_package_with_coremlc(args.input, args.output)