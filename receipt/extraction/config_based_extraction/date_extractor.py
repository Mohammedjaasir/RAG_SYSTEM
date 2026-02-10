import re
import dateparser

def extract_date(clean_text, config):
    """
    Extracts date, cleans OCR errors based on config, and standardizes format.
    """
    # 1. Parse Config
    patterns = []
    target_fmt = "%Y-%m-%d"
    ocr_config = None

    if isinstance(config, list):
        patterns = config # Legacy support
    elif isinstance(config, dict):
        patterns = config.get("patterns", [])
        target_fmt = config.get("target_format", "%Y-%m-%d")
        ocr_config = config.get("ocr_correction")

    # 2. Extract Raw Date String
    raw_date = None
    for pattern in patterns:
        match = re.search(pattern, clean_text, re.IGNORECASE)
        if match:
            raw_date = match.group(0).strip()
            break
    
    if not raw_date:
        return None

    # 3. Dynamic OCR Correction
    clean_raw = raw_date
    if ocr_config:
        trigger = ocr_config.get("trigger_regex", "")
        replacements = ocr_config.get("replacements", {})
        
        # Only apply fixes if the raw string matches the trigger (e.g., contains O/I/l)
        if trigger and re.search(trigger, clean_raw):
            for char_to_find, char_to_replace in replacements.items():
                clean_raw = clean_raw.replace(char_to_find, char_to_replace)

    # 4. Standardization
    try:
        # settings={'DATE_ORDER': 'DMY'} prefers Day-Month-Year (common in UK/EU receipts)
        dt_obj = dateparser.parse(clean_raw, settings={'DATE_ORDER': 'DMY'})
        if dt_obj:
            return dt_obj.strftime(target_fmt)
    except Exception:
        pass

    # Return raw (cleaned) string if parsing fails
    return clean_raw