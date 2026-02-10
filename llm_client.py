"""
Typhoon LLM Client
===================
ใช้ Typhoon v2.5-30b-a3b-instruct (Typhoon Qwen) สำหรับ:
- สรุปเนื้อหาเอกสาร
- เปรียบเทียบและหาจุดต่างระหว่างเอกสาร
- วิเคราะห์ประเภทเอกสาร

หมายเหตุ: การอ่านข้อความ (OCR) ใช้ PaddleOCR ใน ocr_engine.py
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Model constants
TYPHOON_LLM_MODEL = "typhoon-v2.5-30b-a3b-instruct"  # สำหรับสรุปและเปรียบเทียบ
TYPHOON_API_BASE = "https://api.opentyphoon.ai/v1"


class TyphoonClient:
    """
    Client สำหรับ Typhoon LLM (Qwen-based)
    ใช้สำหรับวิเคราะห์ สรุป และเปรียบเทียบเอกสาร
    (ไม่ใช่ OCR - OCR ใช้ PaddleOCR แยกต่างหาก)
    """
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("TYPHOON_API_KEY")
        if not self.api_key:
            print("Warning: TYPHOON_API_KEY not found.")
            self.api_key = "sk-placeholder" 
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=TYPHOON_API_BASE
        )
        # Typhoon Qwen - สำหรับสรุปเนื้อหาและหาจุดต่าง
        self.model = TYPHOON_LLM_MODEL

    def classify_document(self, text, language='en'):
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
                max_tokens=36000,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error connecting to Typhoon API: {e}"

    def summarize_comparison(self, summary_dict, doc1_text="", doc2_text="", language='en'):
        """
        AI Summary - สรุปผลการเปรียบเทียบอย่างละเอียด
        รวมถึงบอกประเภทเอกสาร และรายละเอียดความเหมือน/ต่าง
        """
        mode = summary_dict.get('mode', 'diff')
        doc1_name = summary_dict.get('doc1_name', 'Document 1')
        doc2_name = summary_dict.get('doc2_name', 'Document 2')
        
        # สร้าง stats text
        if mode == 'same':
            matched_doc1 = summary_dict.get('matched_doc1', 0)
            matched_doc2 = summary_dict.get('matched_doc2', 0)
            doc1_unique = summary_dict.get('doc1_unique_words', 1)
            match_rate = (matched_doc1 / doc1_unique * 100) if doc1_unique > 0 else 0
            sample_matched = summary_dict.get('sample_matched', [])
            
            stats_text = f"""
            Mode: Finding MATCHES (หาจุดที่เหมือนกัน)
            Document 1: {doc1_name} ({summary_dict.get('doc1_pages', 0)} pages, {summary_dict.get('doc1_total_words', 0)} words)
            Document 2: {doc2_name} ({summary_dict.get('doc2_pages', 0)} pages, {summary_dict.get('doc2_total_words', 0)} words)
            
            Matching words in Doc1: {matched_doc1}
            Matching words in Doc2: {matched_doc2}
            Match rate: {match_rate:.1f}%
            
            Sample matched words (คำที่ตรงกัน): {', '.join(sample_matched[:30])}
            """
        else:
            diff_doc1 = summary_dict.get('diff_doc1', 0)
            diff_doc2 = summary_dict.get('diff_doc2', 0)
            sample_doc1 = summary_dict.get('sample_doc1', [])
            sample_doc2 = summary_dict.get('sample_doc2', [])
            
            total_unique = summary_dict.get('doc1_unique_words', 0) + summary_dict.get('doc2_unique_words', 0)
            diff_rate = ((diff_doc1 + diff_doc2) / total_unique * 100) if total_unique > 0 else 0
            
            stats_text = f"""
            Mode: Finding DIFFERENCES (หาจุดที่แตกต่าง)
            Document 1: {doc1_name} ({summary_dict.get('doc1_pages', 0)} pages, {summary_dict.get('doc1_total_words', 0)} words)
            Document 2: {doc2_name} ({summary_dict.get('doc2_pages', 0)} pages, {summary_dict.get('doc2_total_words', 0)} words)
            
            Words only in Doc1 (มีเฉพาะในเอกสาร 1): {diff_doc1}
            Words only in Doc2 (มีเฉพาะในเอกสาร 2): {diff_doc2}
            Difference rate: {diff_rate:.1f}%
            
            Sample words only in Doc1: {', '.join(sample_doc1[:20])}
            Sample words only in Doc2: {', '.join(sample_doc2[:20])}
            """
        
        # เพิ่ม sample text จากเอกสาร (ถ้ามี)
        doc_preview = ""
        if doc1_text:
            doc_preview += f"\n--- ตัวอย่างเนื้อหาเอกสาร 1 ---\n{doc1_text[:3000]}\n"
        if doc2_text:
            doc_preview += f"\n--- ตัวอย่างเนื้อหาเอกสาร 2 ---\n{doc2_text[:3000]}\n"
        
        if language == 'th':
            system_msg = """คุณเป็นผู้ช่วย AI ที่เชี่ยวชาญในการวิเคราะห์และสรุปผลการเปรียบเทียบเอกสาร 
            คุณต้องสรุปอย่างละเอียด ครบถ้วน และเป็นประโยชน์"""
            
            prompt = f"""
            จากผลการเปรียบเทียบเอกสาร 2 ฉบับด้านล่าง:
            
            === สถิติการเปรียบเทียบ ===
            {stats_text}
            
            === ตัวอย่างเนื้อหาเอกสาร ===
            {doc_preview}
            
            โปรดสรุปผลการเปรียบเทียบเป็นภาษาไทยอย่างละเอียด โดยต้องครอบคลุม:
            
            1. **ประเภทเอกสาร**: เอกสารทั้งสองฉบับเป็นเอกสารประเภทใด (เช่น ใบเสนอราคา, สัญญา, ใบแจ้งหนี้, กรมธรรม์ประกันภัย, หนังสือราชการ ฯลฯ)
            
            2. **ข้อมูลสำคัญที่พบ**: 
               - ชื่อบุคคล/บริษัท ที่ปรากฏ
               - เลขที่เอกสาร, เลขประจำตัวผู้เสียภาษี
               - วันที่, จำนวนเงิน, ตัวเลขสำคัญ
            
            3. **จุดที่เหมือนกัน**: 
               - ข้อมูลใดบ้างที่ตรงกันทั้งสองฉบับ
               - ระบุตัวอย่างที่ชัดเจน
            
            4. **จุดที่แตกต่างกัน**: 
               - ข้อมูลใดบ้างที่ไม่ตรงกัน
               - ระบุค่าที่ต่างกัน (เช่น เอกสาร 1 ระบุ X แต่เอกสาร 2 ระบุ Y)
            
            5. **สรุปและข้อเสนอแนะ**: 
               - ความแตกต่างมีนัยสำคัญหรือไม่
               - ควรตรวจสอบจุดใดเพิ่มเติม
            
            ตอบเป็นภาษาไทย ใช้รูปแบบที่อ่านง่าย
            """
        else:
            system_msg = """You are an AI assistant specialized in analyzing and summarizing document comparison results.
            You must provide detailed, comprehensive, and useful summaries."""
            
            prompt = f"""
            Based on the document comparison results below:
            
            === Comparison Statistics ===
            {stats_text}
            
            === Document Content Preview ===
            {doc_preview}
            
            Please provide a detailed summary covering:
            
            1. **Document Type**: What type of documents are these? (e.g., invoice, contract, insurance policy, quotation, etc.)
            
            2. **Key Information Found**:
               - Names of persons/companies
               - Document numbers, tax IDs
               - Dates, amounts, important figures
            
            3. **Similarities**: 
               - What information matches between documents?
               - Provide specific examples
            
            4. **Differences**: 
               - What information differs?
               - Specify the different values (e.g., Doc 1 shows X, Doc 2 shows Y)
            
            5. **Conclusion & Recommendations**: 
               - Are the differences significant?
               - What should be verified further?
            
            Format the response clearly and professionally.
            """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=5000,  # เพิ่มขึ้นสำหรับคำตอบละเอียด
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating AI summary: {e}"