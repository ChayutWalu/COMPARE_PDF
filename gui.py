import customtkinter as ctk
from tkinter import filedialog
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
        self.geometry("800x600")

        self.pdf1_path = ""
        self.pdf2_path = ""

        # Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(3, weight=1)
        # Row 4 used to be save button, we might need more space or a popup
        # Let's make the main window bigger or split the view?
        # A scrollable frame for results might be better below the text box.
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
        self.textbox = ctk.CTkTextbox(self, width=760, height=400)
        self.textbox.grid(row=3, column=0, columnspan=2, padx=20, pady=10, sticky="nsew")

        # Save Button
        self.save_btn = ctk.CTkButton(self, text="Save Result", command=self.save_result, state="disabled")
        self.save_btn.grid(row=4, column=0, columnspan=2, padx=20, pady=10)

        # Visual Result Area (Scrollable Frame)
        self.visual_frame = ctk.CTkScrollableFrame(self, label_text="Visual Comparison", height=300)
        self.visual_frame.grid(row=5, column=0, columnspan=2, padx=20, pady=10, sticky="nsew")


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
            
            # Update GUI with images (must be done in main thread really, but let's see if ctk handles it or if I need after)
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
            
            # Image 1
            if img1:
                ctk_img1 = ctk.CTkImage(light_image=img1, dark_image=img1, size=(300, 400)) # Fixed size thumbnail
                img_lbl1 = ctk.CTkLabel(self.visual_frame, image=ctk_img1, text="", cursor="hand2")
                img_lbl1.grid(row=idx*2+1, column=0, padx=10, pady=10)
                img_lbl1.bind("<Button-1>", lambda e, img=img1, t=f"Document 1 - Page {idx+1}": self.open_zoom_window(img, t))
                
                # Add a tooltip or hint (optional, but label below helps)
                hint1 = ctk.CTkLabel(self.visual_frame, text="(Click to Zoom)", font=("Arial", 10))
                hint1.grid(row=idx*2+2, column=0)
            
            # Image 2
            if img2:
                ctk_img2 = ctk.CTkImage(light_image=img2, dark_image=img2, size=(300, 400))
                img_lbl2 = ctk.CTkLabel(self.visual_frame, image=ctk_img2, text="", cursor="hand2")
                img_lbl2.grid(row=idx*2+1, column=1, padx=10, pady=10)
                img_lbl2.bind("<Button-1>", lambda e, img=img2, t=f"Document 2 - Page {idx+1}": self.open_zoom_window(img, t))

                hint2 = ctk.CTkLabel(self.visual_frame, text="(Click to Zoom)", font=("Arial", 10))
                hint2.grid(row=idx*2+2, column=1)

    def open_zoom_window(self, pil_image, title):
        top = ctk.CTkToplevel(self)
        top.title(title)
        top.geometry("1000x800")
        
        # Bring to front
        top.attributes("-topmost", True)
        
        # Scrollable frame for the large image
        scroll_frame = ctk.CTkScrollableFrame(top, label_text=title)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Use a larger size for the image (preserving aspect ratio)
        # 150 dpi is typically around 1240 width for A4.
        # Let's map it to its actual size or slightly limited if super huge.
        w, h = pil_image.size
        
        # Ensure it fits reasonably but is zoomed in
        scale_factor = 1.0 # Display at native resolution (150 dpi is already "zoomed" vs thumbnail)
        
        full_img = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(int(w*scale_factor), int(h*scale_factor)))
        
        lbl = ctk.CTkLabel(scroll_frame, image=full_img, text="")
        lbl.pack(padx=10, pady=10)



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
