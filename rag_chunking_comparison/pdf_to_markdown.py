import pymupdf4llm
import fitz
from pathlib import Path
from typing import Optional

class PDFToMarkdownConverter:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.pdf_name = Path(pdf_path).stem
        
    def convert_to_markdown(self, output_path: Optional[str] = None) -> str:
        md_text = pymupdf4llm.to_markdown(self.pdf_path)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(md_text)
        
        return md_text
    
    def get_pdf_info(self) -> dict:
        doc = fitz.open(self.pdf_path)
        info = {
            'num_pages': len(doc),
            'metadata': doc.metadata,
            'file_size_mb': Path(self.pdf_path).stat().st_size / (1024 * 1024)
        }
        doc.close()
        return info

if __name__ == "__main__":
    pdf_path = "/home/ubuntu/attachments/30af5be2-cb9a-4976-9c52-fcbb02d7b303/CRDS-FY2024-FR-09.pdf"
    converter = PDFToMarkdownConverter(pdf_path)
    
    info = converter.get_pdf_info()
    print(f"PDF Info: {info}")
    
    md_text = converter.convert_to_markdown("output.md")
    print(f"Converted to markdown: {len(md_text)} characters")
