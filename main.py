import sys
import os
import argparse
from ocr_processor import extract_text_from_pdf
from llm_client import TyphoonClient

def process_files(pdf1_path, pdf2_path, progress_callback=None):
    """
    Processes two PDF files and returns the comparison result.
    
    Args:
        pdf1_path (str): Path to the first PDF.
        pdf2_path (str): Path to the second PDF.
        progress_callback (func): Optional callback for status updates (msg).
    
    Returns:
        str: The combined result string.
    """
    output = []
    def log(msg):
        print(msg)
        output.append(msg)
        if progress_callback:
            progress_callback(msg)

    log("Initializing Typhoon Client...")
    try:
        client = TyphoonClient()
    except Exception as e:
        return f"Failed to initialize Typhoon Client: {e}"

    log(f"\nProcessing {pdf1_path}...")
    try:
        text1 = extract_text_from_pdf(pdf1_path)
    except Exception as e:
        return f"Error processing {pdf1_path}: {e}"
        
    log(f"\nProcessing {pdf2_path}...")
    try:
        text2 = extract_text_from_pdf(pdf2_path)
    except Exception as e:
        return f"Error processing {pdf2_path}: {e}"
        
    log("\nClassifying Document 1...")
    classification1 = client.classify_document(text1)
    log(f"--> Result:\n{classification1}\n")
    
    log("Classifying Document 2...")
    classification2 = client.classify_document(text2)
    log(f"--> Result:\n{classification2}\n")
    
    log("Comparing Documents...")
    comparison = client.compare_documents(text1, text2)
    log(f"--> Comparison Result:\n{comparison}\n")
    
    return "\n".join(output)

def main():
    parser = argparse.ArgumentParser(description="PDF Comparison Tool")
    parser.add_argument("pdf1", nargs='?', help="Path to the first PDF file")
    parser.add_argument("pdf2", nargs='?', help="Path to the second PDF file")
    parser.add_argument("--gui", action="store_true", help="Launch the GUI")
    
    args = parser.parse_args()

    if args.gui or (not args.pdf1 and not args.pdf2):
        print("Launching GUI...")
        try:
            import gui
            gui.run_gui()
        except ImportError:
            print("GUI module not found or dependencies missing (customtkinter).")
            print("Run: pip install customtkinter")
    elif args.pdf1 and args.pdf2:
        process_files(args.pdf1, args.pdf2)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

