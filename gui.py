import customtkinter as ctk
from tkinter import filedialog, messagebox
import tkinter as tk
from PIL import ImageTk, Image
import threading
from main import process_files
from ocr_processor import highlight_text_differences
import os

# สำหรับสร้าง PDF Report
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

        self.title("PDF Compare & Classify")
        self.geometry("1000x800")

        self.pdf1_path = ""
        self.pdf2_path = ""
        self.generated_image_pairs = [] # เก็บรูปภาพที่ Gen แล้วไว้ใช้งานต่อ (Export)

        # Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(3, weight=1)
        self.grid_rowconfigure(5, weight=1)

        # PDF 1 Selection
        self.label1 = ctk.CTkLabel(self, text="PDF 1 not selected", fg_color="transparent")
        self.label1.grid(row=0, column=1, padx=20, pady=10, sticky="ew")
        self.btn1 = ctk.CTkButton(self, text="Select PDF 1", command=self.select_pdf1)
        self.btn1.grid(row=0, column=0, padx=20, pady=10)

        # PDF 2 Selection
        self.label2 = ctk.CTkLabel(self, text="PDF 2 not selected")
        self.label2.grid(row=1, column=1, padx=20, pady=10, sticky="ew")
        self.btn2 = ctk.CTkButton(self, text="Select PDF 2", command=self.select_pdf2)
        self.btn2.grid(row=1, column=0, padx=20, pady=10)

        # Run Button
        self.run_btn = ctk.CTkButton(self, text="Compare Documents", command=self.start_processing)
        self.run_btn.grid(row=2, column=0, columnspan=2, padx=20, pady=20)

        # Results Text Area
        self.textbox = ctk.CTkTextbox(self, width=760, height=300)
        self.textbox.grid(row=3, column=0, columnspan=2, padx=20, pady=10, sticky="nsew")

        # Action Buttons Frame
        self.action_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.action_frame.grid(row=4, column=0, columnspan=2, padx=20, pady=10)

        self.save_txt_btn = ctk.CTkButton(self.action_frame, text="Save Text Result", command=self.save_result, state="disabled")
        self.save_txt_btn.pack(side="left", padx=10)
        
        # ปุ่มใหม่: Export PDF Report
        self.export_pdf_btn = ctk.CTkButton(self.action_frame, text="Export PDF Report", command=self.export_pdf_report, state="disabled", fg_color="green")
        self.export_pdf_btn.pack(side="left", padx=10)

        # Visual Result Area
        self.visual_frame = ctk.CTkScrollableFrame(self, label_text="Visual Comparison", height=400)
        self.visual_frame.grid(row=5, column=0, columnspan=2, padx=20, pady=10, sticky="nsew")
        
        self.current_sync_images = [] 

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

        self.run_btn.configure(state="disabled")
        self.export_pdf_btn.configure(state="disabled")
        self.textbox.delete("1.0", "end")
        self.append_text("Starting processing...\n")
        
        thread = threading.Thread(target=self.process_thread)
        thread.start()

    def process_thread(self):
        try:
            result = process_files(self.pdf1_path, self.pdf2_path, progress_callback=self.update_progress)
            self.enable_save()
            
            self.update_progress("Generating visual highlights...")
            # เก็บผลลัพธ์รูปภาพลงตัวแปรของ Class เพื่อใช้ Export
            self.generated_image_pairs = highlight_text_differences(self.pdf1_path, self.pdf2_path)
            
            self.after(0, self.display_images, self.generated_image_pairs)
            self.update_progress("Visual comparison ready.")
            
        except Exception as e:
            self.update_progress(f"An error occurred: {e}")
        finally:
            self.run_btn.configure(state="normal")
            
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
                
                hint1 = ctk.CTkLabel(self.visual_frame, text="(Click to Compare)", font=("Arial", 10))
                hint1.grid(row=idx*2+2, column=0)
            
            if img2:
                ctk_img2 = ctk.CTkImage(light_image=img2, dark_image=img2, size=(300, 400))
                img_lbl2 = ctk.CTkLabel(self.visual_frame, image=ctk_img2, text="", cursor="hand2")
                img_lbl2.grid(row=idx*2+1, column=1, padx=10, pady=10)
                img_lbl2.bind("<Button-1>", on_click)

                hint2 = ctk.CTkLabel(self.visual_frame, text="(Click to Compare)", font=("Arial", 10))
                hint2.grid(row=idx*2+2, column=1)

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
        
        # --- Toolbar ด้านบน ---
        toolbar = ctk.CTkFrame(top, height=40)
        toolbar.pack(fill="x", padx=10, pady=5)
        
        def save_current_view():
            """บันทึกภาพหน้าจอหน้านี้ (รวมซ้าย-ขวา)"""
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")],
                initialfile=f"compare_page_{page_num}.png"
            )
            if file_path:
                try:
                    # สร้างภาพใหม่ขนาดเท่า 2 รูปเรียงกัน
                    total_width = img1.width + img2.width
                    max_height = max(img1.height, img2.height)
                    combined = Image.new('RGB', (total_width, max_height), (255, 255, 255))
                    combined.paste(img1, (0, 0))
                    combined.paste(img2, (img1.width, 0))
                    combined.save(file_path)
                    messagebox.showinfo("Success", f"Saved image to {file_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save image: {e}")

        save_btn = ctk.CTkButton(toolbar, text="💾 Save This Comparison Image", command=save_current_view)
        save_btn.pack(side="right", padx=10)

        # --- ส่วนแสดงผลภาพ (เหมือนเดิม) ---
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
        
        def on_mousewheel(event):
            delta = int(-1*(event.delta/120))
            canvas1.yview_scroll(delta, "units")
            canvas2.yview_scroll(delta, "units")
            return "break"

        scrollbar.configure(command=scroll_both)
        canvas1.configure(yscrollcommand=scrollbar.set)
        
        for c in [canvas1, canvas2]:
            c.bind("<MouseWheel>", on_mousewheel)
            c.bind("<Button-4>", lambda e: on_mousewheel(type('E', (object,), {'delta': 120})()))
            c.bind("<Button-5>", lambda e: on_mousewheel(type('E', (object,), {'delta': -120})()))

        self.current_sync_images = [] 
        def draw_img(canvas, pil_img):
            if not pil_img: return
            tk_img = ImageTk.PhotoImage(pil_img)
            self.current_sync_images.append(tk_img)
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
                self.append_text(f"Result saved to {filename}\n")
            except Exception as e:
                self.append_text(f"Error saving file: {e}\n")

    def export_pdf_report(self):
        """สร้าง Report PDF รวม Text Summary + รูปภาพเปรียบเทียบ"""
        if not self.generated_image_pairs:
            messagebox.showwarning("No Data", "No comparison images available.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF File", "*.pdf")],
            initialfile="Comparison_Report.pdf"
        )
        if not file_path:
            return

        try:
            c = canvas.Canvas(file_path, pagesize=A4)
            width, height = A4
            
            # --- หน้า 1: Summary Text ---
            # พยายามโหลดฟอนต์ไทย (ถ้ามี)
            font_path = "THSarabunNew.ttf" 
            if os.path.exists(font_path):
                pdfmetrics.registerFont(TTFont('THSarabun', font_path))
                c.setFont("THSarabun", 16)
            else:
                c.setFont("Helvetica", 12)
                self.append_text("Warning: THSarabun font not found, Thai text might be broken in PDF.\n")

            c.drawString(20*mm, height - 20*mm, "Comparison Report Summary")
            
            text_content = self.textbox.get("1.0", "end").split('\n')
            y = height - 35*mm
            for line in text_content:
                if y < 20*mm: # ขึ้นหน้าใหม่ถ้าหมดหน้า
                    c.showPage()
                    if os.path.exists(font_path): c.setFont("THSarabun", 16)
                    else: c.setFont("Helvetica", 12)
                    y = height - 20*mm
                
                # กรองตัวอักษรที่ ReportLab อาจไม่รองรับ
                clean_line = line.encode('utf-8', 'ignore').decode('utf-8')
                c.drawString(20*mm, y, clean_line)
                y -= 6*mm # ระยะห่างบรรทัด

            c.showPage() # จบหน้า Text

            # --- หน้าถัดไป: รูปภาพ ---
            for idx, (img1, img2) in enumerate(self.generated_image_pairs):
                # รวมรูปซ้ายขวา
                if img1 and img2:
                    total_w = img1.width + img2.width
                    max_h = max(img1.height, img2.height)
                    combined = Image.new('RGB', (total_w, max_h), (255, 255, 255))
                    combined.paste(img1, (0, 0))
                    combined.paste(img2, (img1.width, 0))
                    
                    # ย่อรูปลงให้พอดีหน้า A4
                    # A4 width approx 595 points, margin 20+20 = 40
                    available_w = width - 40
                    ratio = available_w / float(total_w)
                    target_h = float(max_h) * ratio
                    
                    # ถ้าสูงเกินหน้า A4 ให้ย่ออีก
                    if target_h > (height - 60):
                        ratio = (height - 60) / float(max_h)
                        target_h = float(max_h) * ratio
                        available_w = float(total_w) * ratio

                    # วาดลง PDF
                    c.drawString(20*mm, height - 20*mm, f"Page {idx+1} Comparison")
                    # ReportLab drawImage รับ path หรือ ImageReader
                    from reportlab.lib.utils import ImageReader
                    c.drawImage(ImageReader(combined), 20, height - 30 - target_h, width=available_w, height=target_h)
                    
                    c.showPage()

            c.save()
            messagebox.showinfo("Success", f"Report saved to {file_path}")
            self.append_text(f"Report saved to {file_path}\n")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to export PDF: {e}")
            print(e)

def run_gui():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    run_gui()