import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class TyphoonClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("TYPHOON_API_KEY")
        if not self.api_key:
            raise ValueError("Typhoon API Key is missing. Please set it in .env or pass it to constructor.")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.opentyphoon.ai/v1"
        )
        self.model = "typhoon-v2.5-30b-a3b-instruct"

    def classify_document(self, text, language='en'):
        if language == 'th':
            system_msg = "You are a helpful AI assistant capable of analyzing Thai documents."
            prompt = f"""
            คุณเป็นผู้ช่วย AI ที่เชี่ยวชาญด้านการจัดการเอกสาร
            โปรดวิเคราะห์เนื้อหาของเอกสารต่อไปนี้ แล้วระบุว่ามันคือเอกสารประเภทอะไร (เช่น ใบแจ้งหนี้, สัญญา, กรมธรรม์, ใบสมัคร, ฯลฯ) และสรุปเนื้อหาสำคัญสั้นๆ ให้ด้วย
            
            **คำเตือน:** โปรดตอบเป็นภาษาไทยเท่านั้น
            
            <เนื้อหาเอกสาร>
            {text[:30000]}
            </เนื้อหาเอกสาร>
            
            รูปแบบคำตอบ:
            ประเภทเอกสาร: [ระบุประเภท]
            สรุปย่อ: [สรุปเนื้อหา]
            """
        else:
            system_msg = "You are a helpful AI assistant."
            prompt = f"""
            You are a helpful assistant that classifies documents.
            Please analyze the following document text and provide a classification (e.g., Invoice, Contract, Report, Article, etc.) and a brief summary.
            
            <document_content>
            {text[:30000]}
            </document_content>
            
            Output format:
            Classification: [Type]
            Summary: [Brief Summary]
            """
        
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=4096,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == 2:
                    raise e
                print(f"API call failed (attempt {attempt+1}/3). Retrying in 2s...")
                import time
                time.sleep(2)

    def compare_documents(self, text1, text2, language='en'):
        if language == 'th':
            system_msg = "You are a helpful AI assistant capable of analyzing Thai documents."
            prompt = f"""
            คุณเป็นผู้ช่วย AI ที่เชี่ยวชาญด้านการตรวจสอบและเปรียบเทียบเอกสาร
            โปรดเปรียบเทียบเนื้อหาของเอกสาร 2 ฉบับด้านล่างนี้ และระบุจุดที่เหมือนและจุดที่แตกต่างกันอย่างละเอียด
            
            **คำเตือน:** โปรดตอบเป็นภาษาไทยเท่านั้น
            
            <เอกสารฉบับที่_1>
            {text1[:30000]}
            </เอกสารฉบับที่_1>
            
            <เอกสารฉบับที่_2>
            {text2[:30000]}
            </เอกสารฉบับที่_2>
            
            รูปแบบคำตอบ:
            **จุดที่เหมือนกัน:**
            - [ประเด็นที่ 1]
            - [ประเด็นที่ 2]
            
            **จุดที่แตกต่างกัน:**
            - [ประเด็นที่ 1]
            - [ประเด็นที่ 2]
            
            **สรุปผลการเปรียบเทียบ:**
            [สรุปภาพรวมสั้นๆ ว่าเอกสาร 2 ฉบับนี้มีความสัมพันธ์กันอย่างไร]
            """
        else:
            system_msg = "You are a helpful AI assistant."
            prompt = f"""
            You are a helpful assistant that compares two documents.
            Please compare the following two documents and highlight the key differences and similarities.
            
            <document_1>
            {text1[:30000]}
            </document_1>
            
            <document_2>
            {text2[:30000]}
            </document_2>
            
            Output format:
            **Similarities:**
            - [Point 1]
            
            **Differences:**
            - [Point 1]
            
            **Conclusion:**
            [Brief conclusion]
            """
        
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=8192,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == 2:
                    raise e
                print(f"API call failed (attempt {attempt+1}/3). Retrying in 2s...")
                import time
                time.sleep(2)

if __name__ == "__main__":
    try:
        client = TyphoonClient()
        print("Typhoon Client initialized successfully.")
    except Exception as e:
        print(f"Error initializing client: {e}")