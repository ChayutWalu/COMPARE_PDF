import customtkinter as ctk
from tkinter import filedialog
import tkinter as tk  # เพิ่ม import นี้
from PIL import ImageTk # เพิ่ม import นี้
import threading
from main import process_files
from ocr_processor import highlight_text_differences

import os

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("PDF Compare & Classify")
        self.geometry("1000x800") # ขยายหน้าจอหลัก

        self.pdf1_path = ""
        self.pdf2_path = ""

        # Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(3, weight=1)
        self.grid_rowconfigure(5, weight=1) # Visual diff area

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

        # Save Button
        self.save_btn = ctk.CTkButton(self, text="Save Result", command=self.save_result, state="disabled")
        self.save_btn.grid(row=4, column=0, columnspan=2, padx=20, pady=10)

        # Visual Result Area (Scrollable Frame)
        self.visual_frame = ctk.CTkScrollableFrame(self, label_text="Visual Comparison", height=400)
        self.visual_frame.grid(row=5, column=0, columnspan=2, padx=20, pady=10, sticky="nsew")
        
        # ตัวแปรเก็บภาพเพื่อป้องกัน garbage collection
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
        self.textbox.delete("1.0", "end")
        self.append_text("Starting processing...\n")
        
        # Run in a separate thread to keep GUI responsive
        thread = threading.Thread(target=self.process_thread)
        thread.start()

    def process_thread(self):
        try:
            result = process_files(self.pdf1_path, self.pdf2_path, progress_callback=self.update_progress)
            self.enable_save()
            
            self.update_progress("Generating visual highlights...")
            images = highlight_text_differences(self.pdf1_path, self.pdf2_path)
            
            # Update GUI with images
            self.after(0, self.display_images, images)
            self.update_progress("Visual comparison ready.")
            
        except Exception as e:
            self.update_progress(f"An error occurred: {e}")
            result = ""
        finally:
            self.run_btn.configure(state="normal")
            
    def display_images(self, image_pairs):
        # Clear previous
        for widget in self.visual_frame.winfo_children():
            widget.destroy()

        for idx, (img1, img2) in enumerate(image_pairs):
            # Create a row for this page pair
            
            # Label
            lbl = ctk.CTkLabel(self.visual_frame, text=f"Page {idx+1}")
            lbl.grid(row=idx*2, column=0, columnspan=2, pady=(10, 0))
            
            # สร้างฟังก์ชัน Callback ที่ส่งค่าทั้ง 2 รูปไปพร้อมกัน
            # ใช้ default argument เพื่อ lock ค่าตัวแปรใน loop
            on_click = lambda e, i1=img1, i2=img2, p=idx+1: self.open_sync_window(i1, i2, p)

            # Image 1 (Thumbnail)
            if img1:
                ctk_img1 = ctk.CTkImage(light_image=img1, dark_image=img1, size=(300, 400))
                img_lbl1 = ctk.CTkLabel(self.visual_frame, image=ctk_img1, text="", cursor="hand2")
                img_lbl1.grid(row=idx*2+1, column=0, padx=10, pady=10)
                img_lbl1.bind("<Button-1>", on_click)
                
                hint1 = ctk.CTkLabel(self.visual_frame, text="(Click to Compare)", font=("Arial", 10))
                hint1.grid(row=idx*2+2, column=0)
            
            # Image 2 (Thumbnail)
            if img2:
                ctk_img2 = ctk.CTkImage(light_image=img2, dark_image=img2, size=(300, 400))
                img_lbl2 = ctk.CTkLabel(self.visual_frame, image=ctk_img2, text="", cursor="hand2")
                img_lbl2.grid(row=idx*2+1, column=1, padx=10, pady=10)
                img_lbl2.bind("<Button-1>", on_click)

                hint2 = ctk.CTkLabel(self.visual_frame, text="(Click to Compare)", font=("Arial", 10))
                hint2.grid(row=idx*2+2, column=1)

    def open_sync_window(self, img1, img2, page_num):
        """หน้าต่างเปรียบเทียบแบบ Sync Scroll"""
        top = ctk.CTkToplevel(self)
        top.title(f"Comparison View - Page {page_num} (Synchronized Scrolling)")
        top.geometry("1400x900")
        top.attributes("-topmost", True)
        
        # 1. Main Container
        container = ctk.CTkFrame(top)
        container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 2. Scrollbar (ใช้ควบคุมทั้ง 2 ฝั่ง)
        scrollbar = ctk.CTkScrollbar(container, orientation="vertical")
        scrollbar.pack(side="right", fill="y")
        
        # 3. Canvas Setup (ใช้ tk.Canvas เพื่อการควบคุมที่ละเอียดกว่า ctkScrollableFrame)
        # แบ่งครึ่งซ้ายขวา
        pane = tk.PanedWindow(container, orient="horizontal", sashwidth=5, bg="#404040")
        pane.pack(fill="both", expand=True, side="left")

        canvas1 = tk.Canvas(pane, bg="#303030", highlightthickness=0)
        canvas2 = tk.Canvas(pane, bg="#303030", highlightthickness=0)
        
        pane.add(canvas1)
        pane.add(canvas2)
        
        # 4. Sync Logic Function
        def scroll_both(*args):
            # เมื่อ Scrollbar ขยับ -> สั่ง Canvas ทั้งคู่ขยับตาม
            canvas1.yview(*args)
            canvas2.yview(*args)
        
        def on_mousewheel(event):
            # เมื่อหมุนเมาส์ -> สั่ง Scrollbar และ Canvas ขยับ
            # Windows: event.delta, Mac/Linux อาจต่างกันเล็กน้อย
            delta = int(-1*(event.delta/120))
            canvas1.yview_scroll(delta, "units")
            canvas2.yview_scroll(delta, "units")
            return "break" # ป้องกันการทำงานซ้ำซ้อน

        # เชื่อม Scrollbar เข้ากับฟังก์ชัน Sync
        scrollbar.configure(command=scroll_both)
        
        # เชื่อม Canvas กลับไปหา Scrollbar (เอาแค่ฝั่งซ้ายเป็น Master ก็พอ)
        canvas1.configure(yscrollcommand=scrollbar.set)
        
        # Bind MouseWheel
        canvas1.bind("<MouseWheel>", on_mousewheel)
        canvas2.bind("<MouseWheel>", on_mousewheel)
        # Linux compatibility (Button-4/5)
        canvas1.bind("<Button-4>", lambda e: on_mousewheel(type('Event', (object,), {'delta': 120})()))
        canvas1.bind("<Button-5>", lambda e: on_mousewheel(type('Event', (object,), {'delta': -120})()))

        # 5. Draw Images
        self.current_sync_images = [] # Clear references

        def draw_img(canvas, pil_img):
            if not pil_img: return
            # แปลงเป็น PhotoImage สำหรับ tkinter
            tk_img = ImageTk.PhotoImage(pil_img)
            self.current_sync_images.append(tk_img) # ต้องเก็บ ref ไว้ไม่งั้นภาพหาย
            
            # วาดภาพลง Canvas
            canvas.create_image(0, 0, anchor="nw", image=tk_img)
            
            # ตั้งค่า Scroll Region ให้เท่ากับขนาดภาพ
            canvas.configure(scrollregion=(0, 0, pil_img.width, pil_img.height))

        draw_img(canvas1, img1)
        draw_img(canvas2, img2)


    def update_progress(self, msg):
        self.textbox.insert("end", msg + "\n")
        self.textbox.see("end")

    def append_text(self, text):
        self.textbox.insert("end", text)
        self.textbox.see("end")

    def enable_save(self):
        self.save_btn.configure(state="normal")
        
    def save_result(self):
        filename = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        if filename:
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(self.textbox.get("1.0", "end"))
                self.append_text(f"Result saved to {filename}\n")
            except Exception as e:
                self.append_text(f"Error saving file: {e}\n")

def run_gui():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    run_gui()