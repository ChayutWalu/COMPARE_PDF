import sys
import os
import argparse
from ocr_processor_paddle import extract_text_from_pdf
from llm_client import TyphoonClient

def process_files(pdf1_path, pdf2_path, language='en', mode='diff', progress_callback=None):
    """
    Process PDFs with mode selection ('diff' or 'same')
    """
    output = []
    def log(msg):
        print(msg)
        output.append(msg)
        if progress_callback:
            progress_callback(msg)

    log("Initializing...")
    log("  → OCR: Typhoon OCR 1.5 via Ollama (LOCAL - ข้อมูลไม่ออกนอกเครื่อง)")
    log("  → LLM: Typhoon Qwen v2.5-30b via API (สรุปและเปรียบเทียบ)")
    client = TyphoonClient()

    log(f"\n[LOCAL OCR] Processing {pdf1_path}...")
    try:
        text1, engine1 = extract_text_from_pdf(pdf1_path)
        log(f"  ✅ อ่านข้อความด้วย: {'Typhoon OCR 1.5 (Local)' if engine1 == 'typhoon' else 'EasyOCR (fallback)'}")
    except Exception as e:
        return f"Error processing {pdf1_path}: {e}"
        
    log(f"\n[LOCAL OCR] Processing {pdf2_path}...")
    try:
        text2, engine2 = extract_text_from_pdf(pdf2_path)
        log(f"  ✅ อ่านข้อความด้วย: {'Typhoon OCR 1.5 (Local)' if engine2 == 'typhoon' else 'EasyOCR (fallback)'}")
    except Exception as e:
        return f"Error processing {pdf2_path}: {e}"
    
    log(f"\n[API] Comparing with Typhoon Qwen (Mode: {mode.upper()})...")
    comparison = client.compare_documents(text1, text2, language=language, mode=mode)
    log(f"--> Comparison Result:\n{comparison}\n")
    
    return "\n".join(output)

def main():
    parser = argparse.ArgumentParser(description="PDF Comparison Tool")
    parser.add_argument("pdf1", nargs='?', help="Path to first PDF")
    parser.add_argument("pdf2", nargs='?', help="Path to second PDF")
    parser.add_argument("--mode", default="diff", choices=["diff", "same"], help="Comparison mode")
    parser.add_argument("--gui", action="store_true", help="Launch GUI")
    
    args = parser.parse_args()

    if args.gui or (not args.pdf1 and not args.pdf2):
        try:
            import gui
            gui.run_gui()
        except ImportError as e:
            print(f"GUI Error: {e}")
            print("Run: pip install customtkinter")
    elif args.pdf1 and args.pdf2:
        print(process_files(args.pdf1, args.pdf2, mode=args.mode))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()