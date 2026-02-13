import streamlit as st
import os
import sys
import json
import base64
from pathlib import Path
from PIL import Image
import tempfile

# Add local directories to path for imports
root_path = Path(__file__).parent.absolute()
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / "receipt" / "rag"))

try:
    from receipt.rag.ocr_client import get_ocr_client
    from receipt.rag.receipt_rag import get_rag_pipeline
except ImportError:
    # Fallback for different path structures
    from ocr_client import get_ocr_client
    from receipt_rag import get_rag_pipeline

# Page Config
st.set_page_config(
    page_title="Receipt RAG Pro",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .hallucination-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    st.info("Connects to OCR service (8001) and Ollama (11434)")
    
    ollama_model = st.selectbox(
        "Ollama Model",
        ["phi3.5", "llama3", "mistral"],
        index=0
    )
    
    st.divider()
    
    if st.button("🧹 Clear Vector Database", use_container_width=True):
        with st.spinner("Cleaning database..."):
            pipeline = get_rag_pipeline()
            pipeline.clean_database()
            st.success("Database cleared!")

# Main UI
st.title("🧾 Receipt RAG Pro")
st.markdown("Upload a receipt image or PDF to extract structured data with Hallucination Prevention.")

uploaded_file = st.file_uploader("Choose a receipt image...", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("🖼️ Preview")
        # Display image
        if uploaded_file.type == "application/pdf":
            st.warning("PDF preview not supported in this view, but processing will work!")
        else:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            
    with col2:
        st.subheader("🧩 Extraction Results")
        
        # Save to temp file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
            
        try:
            # 1. OCR Step
            with st.status("🔍 Processing OCR...", expanded=False) as status:
                ocr_client = get_ocr_client()
                st.write(f"Connecting to OCR service at {ocr_client.base_url}...")
                ocr_result = ocr_client.process_image(tmp_path)
                st.write(f"Extracted {len(ocr_result.text)} characters.")
                status.update(label="✅ OCR Complete!", state="complete")
            
            # 2. RAG Step
            with st.status("🧠 Running RAG Extraction...", expanded=True) as status:
                st.write("Initializing RAG pipeline...")
                rag_pipeline = get_rag_pipeline(model_name=ollama_model)
                st.write(f"Extracting fields using {ollama_model}...")
                rag_result = rag_pipeline.extract_from_ocr(ocr_result.text)
                status.update(label="✅ Extraction Complete!", state="complete")
                
            # Results Display
            data = rag_result.extracted_data
            
            # Confidence Metric
            conf_color = "normal" if rag_result.confidence > 0.8 else "inverse"
            st.metric("Overall Confidence", f"{rag_result.confidence:.2%}", delta_color=conf_color)
            
            # Hallucination Warnings
            if rag_result.hallucination_report:
                st.markdown('<div class="hallucination-warning">', unsafe_allow_html=True)
                st.markdown("### ⚠️ Hallucination Warnings")
                for warning in rag_result.hallucination_report:
                    st.write(f"❌ {warning}")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.success("✅ Data grounded in OCR text (No hallucinations detected)")
            
            # Horizontal Metrics
            def get_v(k):
                val = data.get(k, 'N/A')
                if isinstance(val, dict) and 'value' in val:
                    return val['value']
                return val
            
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Supplier", get_v('supplier_name'))
            with m2:
                date_raw = get_v('date')
                date_str = date_raw[0].get('value') if isinstance(date_raw, list) and date_raw and isinstance(date_raw[0], dict) else date_raw
                st.metric("Date", str(date_str))
            with m3:
                st.metric("Total Amount", f"{get_v('total_amount')}")
                
            # Detailed Info
            with st.expander("📝 Detailed Information", expanded=True):
                st.write(f"**Address:** {get_v('address')}")
                st.write(f"**Receipt #:** {get_v('receipt_number')}")
                st.write(f"**VAT Amount:** {get_v('vat_amount') or get_v('total_tax_amount')}")
                
            # Items Table
            st.subheader("🛒 Line Items")
            items = data.get('items', [])
            if items:
                import pandas as pd
                item_rows = []
                for item in items:
                    if isinstance(item, dict):
                        qty = item.get('quantity', [])
                        qty = qty[0] if isinstance(qty, list) and qty else '1'
                        name = item.get('name', ['N/A'])
                        name = name[0] if isinstance(name, list) and name else 'N/A'
                        price = item.get('total_price', {})
                        price_val = price.get('value', 'N/A') if isinstance(price, dict) else price
                        item_rows.append({
                            "Qty": qty,
                            "Description": name,
                            "Price": price_val
                        })
                st.table(pd.DataFrame(item_rows))
            else:
                st.info("No line items extracted.")
                
            # Raw Data Tabs
            t1, t2 = st.tabs(["📄 Raw JSON", "🔡 OCR Text"])
            with t1:
                st.json(data)
            with t2:
                st.text_area("Full OCR Text", ocr_result.text, height=300)
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.exception(e)
        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
else:
    # Landing state
    st.info("Please upload a receipt image to begin extraction.")
    
    st.markdown("""
    ### How it works:
    1. **OCR Step:** The image is sent to a high-accuracy OCR service to extract raw text and layout.
    2. **RAG Retrieval:** Similar receipt patterns are retrieved from the knowledge base to guide the LLM.
    3. **LLM Extraction:** Phi-3.5 or Llama-3 extracts structured fields from the text.
    4. **Verification:** A specialized layer checks for hallucinations and context leakage.
    """)
