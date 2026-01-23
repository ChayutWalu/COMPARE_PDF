import customtkinter as ctk
from tkinter import filedialog, messagebox
import tkinter as tk
from PIL import ImageTk, Image
import threading
from main import process_files
from ocr_processor import highlight_text_differences
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("PDF Compare (Diff/Match)")
        self.geometry("1150x900")

        self.pdf1_path = ""
        self.pdf2_path = ""
        self.generated_image_pairs = []
        self.comparison_summary = {}  # เก็บ summary

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(5, weight=1) 
        self.grid_rowconfigure(7, weight=1)

        # PDF 1
        self.label1 = ctk.CTkLabel(self, text="PDF 1 not selected", fg_color="transparent")
        self.label1.grid(row=0, column=1, padx=20, pady=10, sticky="ew")
        self.btn1 = ctk.CTkButton(self, text="Select PDF 1", command=self.select_pdf1)
        self.btn1.grid(row=0, column=0, padx=20, pady=10)

        # PDF 2
        self.label2 = ctk.CTkLabel(self, text="PDF 2 not selected")
        self.label2.grid(row=1, column=1, padx=20, pady=10, sticky="ew")
        self.btn2 = ctk.CTkButton(self, text="Select PDF 2", command=self.select_pdf2)
        self.btn2.grid(row=1, column=0, padx=20, pady=10)

        # --- Settings Frame (Language & Mode) ---
        self.settings_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.settings_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Language
        self.lang_label = ctk.CTkLabel(self.settings_frame, text="Language:")
        self.lang_label.pack(side="left", padx=(0, 10))
        self.lang_combobox = ctk.CTkComboBox(self.settings_frame, values=["English", "Thai (ภาษาไทย)"], state="readonly", width=140)
        self.lang_combobox.pack(side="left", padx=(0, 20))
        self.lang_combobox.set("Thai (ภาษาไทย)")

        # Mode Selection
        self.mode_label = ctk.CTkLabel(self.settings_frame, text="Mode:")
        self.mode_label.pack(side="left", padx=(0, 10))
        self.mode_var = ctk.StringVar(value="diff")
        self.mode_switch = ctk.CTkSegmentedButton(self.settings_frame, values=["Find Differences", "Find Matches"], variable=self.mode_var)
        self.mode_switch.pack(side="left")
        self.mode_switch.set("Find Differences")

        # Run Button
        self.run_btn = ctk.CTkButton(self, text="Compare Documents", command=self.start_processing, height=40, font=("Arial", 14, "bold"))
        self.run_btn.grid(row=3, column=0, columnspan=2, padx=20, pady=15)

        # Progress Bar Frame
        self.progress_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.progress_frame.grid(row=4, column=0, columnspan=2, padx=20, pady=5, sticky="ew")
        
        self.progress_label = ctk.CTkLabel(self.progress_frame, text="Ready", anchor="w")
        self.progress_label.pack(fill="x", padx=5)
        
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame, height=20)
        self.progress_bar.pack(fill="x", padx=5, pady=5)
        self.progress_bar.set(0)

        # Text Output
        self.textbox = ctk.CTkTextbox(self, width=760, height=180) 
        self.textbox.grid(row=5, column=0, columnspan=2, padx=20, pady=5, sticky="nsew")

        # Action Buttons
        self.action_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.action_frame.grid(row=6, column=0, columnspan=2, padx=20, pady=5)

        self.save_txt_btn = ctk.CTkButton(self.action_frame, text="Save Text Result", command=self.save_result, state="disabled")
        self.save_txt_btn.pack(side="left", padx=10)
        
        self.export_pdf_btn = ctk.CTkButton(self.action_frame, text="Export PDF Report", command=self.export_pdf_report, state="disabled", fg_color="green")
        self.export_pdf_btn.pack(side="left", padx=10)

        # Visual Result
        self.visual_frame = ctk.CTkScrollableFrame(self, label_text="Visual Comparison", height=300)
        self.visual_frame.grid(row=7, column=0, columnspan=2, padx=20, pady=10, sticky="nsew")

    def select_pdf1(self):
        filename = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if filename:
            self.pdf1_path = filename
            self.label1.configure(text=os.path.basename(filename))

    def select_pdf2(self):
        filename = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if filename:
            self.pdf2_path = filename
            self.label2.configure(text=os.path.basename(filename))

    def start_processing(self):
        if not self.pdf1_path or not self.pdf2_path:
            self.append_text("Please select both PDF files.\n")
            return

        selected_lang_str = self.lang_combobox.get()
        lang_code = 'th' if "Thai" in selected_lang_str else 'en'
        
        mode_ui = self.mode_var.get()
        mode_val = 'same' if mode_ui == "Find Matches" else 'diff'

        self.run_btn.configure(state="disabled")
        self.export_pdf_btn.configure(state="disabled")
        self.textbox.delete("1.0", "end")
        self.progress_bar.set(0)
        self.progress_label.configure(text="Starting...")
        self.append_text(f"Starting processing... (Mode: {mode_ui})\n")
        
        thread = threading.Thread(target=self.process_thread, args=(lang_code, mode_val))
        thread.start()

    def process_thread(self, lang_code, mode_val):
        try:
            # Reset progress
            self.after(0, lambda: self.progress_bar.set(0))
            self.after(0, lambda: self.progress_label.configure(text="Starting..."))
            
            # 1. ส่ง mode ไปที่ LLM
            result = process_files(self.pdf1_path, self.pdf2_path, language=lang_code, mode=mode_val, progress_callback=self.update_progress)
            self.enable_save()
            
            # 2. ส่ง mode ไปที่ Image Highlight พร้อม progress callback
            self.update_progress("\nGenerating visual highlights...")
            self.after(0, lambda: self.progress_label.configure(text="Generating visual highlights..."))
            
            def visual_progress(msg, percent=None):
                self.update_progress(msg)
                if percent is not None:
                    self.after(0, lambda p=percent: self.progress_bar.set(p / 100))
                    self.after(0, lambda m=msg: self.progress_label.configure(text=m))
            
            result = highlight_text_differences(self.pdf1_path, self.pdf2_path, mode=mode_val, progress_callback=visual_progress)
            
            # รองรับทั้ง return แบบเก่า (list) และแบบใหม่ (tuple)
            if isinstance(result, tuple):
                self.generated_image_pairs, self.comparison_summary = result
            else:
                self.generated_image_pairs = result
                self.comparison_summary = {}
            
            # แสดง Summary ใน textbox
            if self.comparison_summary:
                self.after(0, self.display_summary, self.comparison_summary)
            
            self.after(0, self.display_images, self.generated_image_pairs)
            self.update_progress("\n✅ Visual comparison ready!")
            self.after(0, lambda: self.progress_bar.set(1))
            self.after(0, lambda: self.progress_label.configure(text="✅ Complete!"))
            
        except Exception as e:
            self.update_progress(f"An error occurred: {e}")
            self.after(0, lambda: self.progress_label.configure(text=f"❌ Error: {e}"))
            import traceback
            traceback.print_exc()
        finally:
            self.after(0, lambda: self.run_btn.configure(state="normal"))

    def display_summary(self, summary):
        """แสดง Summary ใน textbox"""
        self.append_text("\n" + "="*60 + "\n")
        self.append_text("📊 COMPARISON SUMMARY\n")
        self.append_text("="*60 + "\n")
        
        mode_text = "Finding MATCHES" if summary.get('mode') == 'same' else "Finding DIFFERENCES"
        self.append_text(f"Mode: {mode_text}\n")
        self.append_text("-"*60 + "\n")
        
        self.append_text(f"📄 Document 1: {summary.get('doc1_name', 'N/A')}\n")
        self.append_text(f"   Pages: {summary.get('doc1_pages', 0)}, Words: {summary.get('doc1_total_words', 0)}, Unique: {summary.get('doc1_unique_words', 0)}\n")
        
        self.append_text(f"📄 Document 2: {summary.get('doc2_name', 'N/A')}\n")
        self.append_text(f"   Pages: {summary.get('doc2_pages', 0)}, Words: {summary.get('doc2_total_words', 0)}, Unique: {summary.get('doc2_unique_words', 0)}\n")
        
        self.append_text("-"*60 + "\n")
        
        if summary.get('mode') == 'same':
            self.append_text(f"✅ Matching words in Doc1: {summary.get('matched_doc1', 0)}\n")
            self.append_text(f"✅ Matching words in Doc2: {summary.get('matched_doc2', 0)}\n")
            
            if summary.get('doc1_unique_words', 0) > 0:
                match_pct = (summary.get('matched_doc1', 0) / summary.get('doc1_unique_words', 1)) * 100
                self.append_text(f"📈 Match rate: {match_pct:.1f}%\n")
        else:
            self.append_text(f"🔴 Words only in Doc1: {summary.get('diff_doc1', 0)}\n")
            self.append_text(f"🟢 Words only in Doc2: {summary.get('diff_doc2', 0)}\n")
            
            sample1 = summary.get('sample_doc1', [])[:5]
            sample2 = summary.get('sample_doc2', [])[:5]
            if sample1:
                self.append_text(f"📝 Sample Doc1: {sample1}\n")
            if sample2:
                self.append_text(f"📝 Sample Doc2: {sample2}\n")
            
            total_unique = summary.get('doc1_unique_words', 0) + summary.get('doc2_unique_words', 0)
            if total_unique > 0:
                diff_count = summary.get('diff_doc1', 0) + summary.get('diff_doc2', 0)
                diff_pct = (diff_count / total_unique) * 100
                self.append_text(f"📈 Difference rate: {diff_pct:.1f}%\n")
        
        self.append_text("="*60 + "\n")
            
    def display_images(self, image_pairs):
        for widget in self.visual_frame.winfo_children():
            widget.destroy()

        for idx, (img1, img2) in enumerate(image_pairs):
            lbl = ctk.CTkLabel(self.visual_frame, text=f"Page {idx+1}")
            lbl.grid(row=idx*2, column=0, columnspan=2, pady=(10, 0))
            
            on_click = lambda e, i1=img1, i2=img2, p=idx+1: self.open_sync_window(i1, i2, p)

            if img1:
                ctk_img1 = ctk.CTkImage(light_image=img1, dark_image=img1, size=(300, 400))
                img_lbl1 = ctk.CTkLabel(self.visual_frame, image=ctk_img1, text="", cursor="hand2")
                img_lbl1.grid(row=idx*2+1, column=0, padx=10, pady=10)
                img_lbl1.bind("<Button-1>", on_click)
            
            if img2:
                ctk_img2 = ctk.CTkImage(light_image=img2, dark_image=img2, size=(300, 400))
                img_lbl2 = ctk.CTkLabel(self.visual_frame, image=ctk_img2, text="", cursor="hand2")
                img_lbl2.grid(row=idx*2+1, column=1, padx=10, pady=10)
                img_lbl2.bind("<Button-1>", on_click)

    def open_sync_window(self, img1, img2, page_num):
        if not img1 and not img2: return

        top = ctk.CTkToplevel(self)
        top.title(f"Comparison View - Page {page_num}")
        
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.9)
        top.geometry(f"{window_width}x{window_height}")
        top.attributes("-topmost", True)
        
        toolbar = ctk.CTkFrame(top, height=40)
        toolbar.pack(fill="x", padx=10, pady=5)
        
        def save_current_view():
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png")],
                initialfile=f"compare_page_{page_num}.png"
            )
            if file_path:
                try:
                    total_width = img1.width + img2.width
                    max_height = max(img1.height, img2.height)
                    combined = Image.new('RGB', (total_width, max_height), (255, 255, 255))
                    combined.paste(img1, (0, 0))
                    combined.paste(img2, (img1.width, 0))
                    combined.save(file_path)
                    messagebox.showinfo("Success", f"Saved image to {file_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save image: {e}")

        save_btn = ctk.CTkButton(toolbar, text="Save Image", command=save_current_view)
        save_btn.pack(side="right", padx=10)

        target_width = (window_width // 2) - 40 
        def resize_to_fit(pil_img, target_w):
            if not pil_img: return None
            w_percent = (target_w / float(pil_img.size[0]))
            h_size = int((float(pil_img.size[1]) * float(w_percent)))
            return pil_img.resize((target_w, h_size), Image.Resampling.LANCZOS)

        resized_img1 = resize_to_fit(img1, target_width)
        resized_img2 = resize_to_fit(img2, target_width)

        container = ctk.CTkFrame(top)
        container.pack(fill="both", expand=True, padx=10, pady=10)
        
        scrollbar = ctk.CTkScrollbar(container, orientation="vertical")
        scrollbar.pack(side="right", fill="y")
        
        content_frame = tk.Frame(container, bg="#303030")
        content_frame.pack(side="left", fill="both", expand=True)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=1)
        content_frame.grid_rowconfigure(0, weight=1)

        canvas1 = tk.Canvas(content_frame, bg="#404040", highlightthickness=0)
        canvas1.grid(row=0, column=0, sticky="nsew", padx=(0, 2))
        
        canvas2 = tk.Canvas(content_frame, bg="#404040", highlightthickness=0)
        canvas2.grid(row=0, column=1, sticky="nsew", padx=(2, 0))

        def scroll_both(*args):
            canvas1.yview(*args)
            canvas2.yview(*args)
        
        scrollbar.configure(command=scroll_both)
        canvas1.configure(yscrollcommand=scrollbar.set)
        
        def on_mousewheel(event):
            delta = int(-1*(event.delta/120))
            canvas1.yview_scroll(delta, "units")
            canvas2.yview_scroll(delta, "units")
            return "break"
        
        for c in [canvas1, canvas2]:
            c.bind("<MouseWheel>", on_mousewheel)

        def draw_img(canvas, pil_img):
            if not pil_img: return
            tk_img = ImageTk.PhotoImage(pil_img)
            if not hasattr(top, 'tk_images'): top.tk_images = []
            top.tk_images.append(tk_img)
            canvas.create_image(0, 0, anchor="nw", image=tk_img)
            canvas.configure(scrollregion=(0, 0, pil_img.width, pil_img.height))

        draw_img(canvas1, resized_img1)
        draw_img(canvas2, resized_img2)

    def update_progress(self, msg):
        self.textbox.insert("end", msg + "\n")
        self.textbox.see("end")

    def append_text(self, text):
        self.textbox.insert("end", text)
        self.textbox.see("end")

    def enable_save(self):
        self.save_txt_btn.configure(state="normal")
        self.export_pdf_btn.configure(state="normal")
        
    def save_result(self):
        filename = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        if filename:
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(self.textbox.get("1.0", "end"))
            except Exception as e:
                pass

    def export_pdf_report(self):
        if not self.generated_image_pairs: return
        file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF File", "*.pdf")])
        if not file_path: return
        try:
            c = canvas.Canvas(file_path, pagesize=A4)
            width, height = A4
            font_path = "THSarabunNew.ttf" 
            if os.path.exists(font_path):
                pdfmetrics.registerFont(TTFont('THSarabun', font_path))
                c.setFont("THSarabun", 16)
            else:
                c.setFont("Helvetica", 12)
            
            c.drawString(20*mm, height - 20*mm, f"Comparison Report (Mode: {self.mode_var.get()})")
            c.showPage()
            c.save()
            messagebox.showinfo("Success", "Saved")
        except Exception as e:
            messagebox.showerror("Error", f"{e}")

def run_gui():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    run_gui()