import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class TyphoonClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("TYPHOON_API_KEY")
        if not self.api_key:
            # ใช้ Key หลอกเพื่อป้องกัน Error หากยังไม่มี env (แต่จะใช้จริงไม่ได้)
            print("Warning: TYPHOON_API_KEY not found.")
            self.api_key = "sk-placeholder" 
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.opentyphoon.ai/v1"
        )
        self.model = "typhoon-v2.5-30b-a3b-instruct"

    def classify_document(self, text, language='en'):
        # (คงเดิม - ไม่ได้เปลี่ยนแปลงส่วนนี้)
        if language == 'th':
            system_msg = "You are a helpful AI assistant capable of analyzing Thai documents."
            prompt = f"โปรดวิเคราะห์เอกสารและสรุปสั้นๆ: \n\n{text[:10000]}"
        else:
            system_msg = "You are a helpful AI assistant."
            prompt = f"Analyze and summarize this document: \n\n{text[:10000]}"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"

    def compare_documents(self, text1, text2, language='en', mode='diff'):
        """
        compare_documents พร้อมรองรับ mode ('diff' หรือ 'same')
        """
        if language == 'th':
            system_msg = "You are a helpful AI assistant capable of analyzing Thai documents."
            
            # ปรับ Prompt ตามโหมด
            if mode == 'same':
                focus_instruction = "**เน้นระบุเฉพาะจุดที่เหมือนกัน (Similarities)** ของข้อมูลสำคัญ (เช่น ชื่อ, เลขที่, วันที่, จำนวนเงิน) ไม่ต้องกล่าวถึงจุดที่ต่างมากนัก"
            else:
                focus_instruction = "**เน้นระบุจุดที่แตกต่างกัน (Differences)** อย่างละเอียด เช่น คำผิด, ตัวเลขที่ไม่ตรงกัน, หรือข้อความที่หายไป"

            prompt = f"""
            คุณเป็นผู้ช่วย AI ที่เชี่ยวชาญด้านการตรวจสอบเอกสาร
            โปรดเปรียบเทียบเนื้อหาของเอกสาร 2 ฉบับด้านล่างนี้ และ {focus_instruction}
            
            **คำเตือน:** โปรดตอบเป็นภาษาไทยเท่านั้น
            
            <เอกสารฉบับที่_1>
            {text1[:20000]}
            </เอกสารฉบับที่_1>
            
            <เอกสารฉบับที่_2>
            {text2[:20000]}
            </เอกสารฉบับที่_2>
            """
        else:
            system_msg = "You are a helpful AI assistant."
            if mode == 'same':
                focus_instruction = "Focus primarily on identifying **Similarities** between the two documents."
            else:
                focus_instruction = "Focus primarily on identifying **Differences** between the two documents."

            prompt = f"""
            Compare the following two documents. {focus_instruction}
            
            <document_1>
            {text1[:20000]}
            </document_1>
            
            <document_2>
            {text2[:20000]}
            </document_2>
            """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=36000, # ลดลงหน่อยเพื่อความเร็ว
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error connecting to Typhoon API: {e}"