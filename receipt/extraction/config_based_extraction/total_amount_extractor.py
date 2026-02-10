import re

def extract_total(lines, rules):
    """
    Scans lines bottom-to-top.
    Skips lines matching 'excludes'.
    Finds first line with a keyword and valid regex match.
    """
    keywords = rules.get('keywords', [])
    excludes = rules.get('excludes', []) # <--- NEW: Generic Exclusion Support
    regex_config = rules.get('regex', [])
    
    patterns = regex_config if isinstance(regex_config, list) else [regex_config]
    
    for line in reversed(lines):
        line_lower = line.lower()
        
        # 1. EXCLUSION CHECK (The Fix)
        # If the line contains "change due" or "change", skip it entirely.
        if any(ex in line_lower for ex in excludes):
            continue
            
        # 2. Keyword Check
        if any(k in line_lower for k in keywords):
            
            best_amount = None
            
            for pattern in patterns:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                
                for m in matches:
                    try:
                        clean_str = ""
                        # Handle Capture Groups (16 ... 00)
                        if m.groups():
                            valid_groups = [g for g in m.groups() if g]
                            clean_str = ".".join(valid_groups)
                        # Handle Single Match (5.50)
                        else:
                            clean_str = m.group(0).replace(',', '.')

                        # Clean Noise
                        clean_str = re.sub(r"[^\d.]", "", clean_str)
                        val = float(clean_str)
                        
                        if best_amount is None or val > best_amount[0]:
                            best_amount = (val, f"{val:.2f}")

                    except ValueError:
                        continue
            
            if best_amount:
                return best_amount[1]
                
    return None